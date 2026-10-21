import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.matching import DSSM, SASRec
from torch_rechub.trainers import CTRTrainer, MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gather_inbatch_logits, gen_model_input, generate_seq_feature_match, inbatch_negative_sampling


def test_inbatch_negative_sampling_random_and_uniform():
    scores = torch.zeros((4, 4))
    neg_idx = inbatch_negative_sampling(scores, neg_ratio=2, generator=torch.Generator().manual_seed(0))
    logits = gather_inbatch_logits(scores, neg_idx)
    assert logits.shape == (4, 3)
    assert neg_idx.shape == (4, 2)
    for row, sampled in enumerate(neg_idx):
        assert row not in sampled.tolist()

    # Different seed should give different permutations to ensure randomness
    neg_idx_second = inbatch_negative_sampling(scores, neg_ratio=2, generator=torch.Generator().manual_seed(1))
    assert not torch.equal(neg_idx, neg_idx_second)


def test_inbatch_negative_sampling_hard_negative():
    scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 0.0]])
    neg_idx = inbatch_negative_sampling(scores, neg_ratio=1, hard_negative=True)
    # highest non-diagonal scores for each row
    assert torch.equal(neg_idx.squeeze(1), torch.tensor([2, 2, 1]))


def _build_small_match_dataloader():
    np.random.seed(42)
    n_users, n_items, n_samples = 12, 24, 80
    data = pd.DataFrame({
        "user_id": np.random.randint(0,
                                     n_users,
                                     n_samples),
        "item_id": np.random.randint(0,
                                     n_items,
                                     n_samples),
        "time": np.arange(n_samples),
    })
    user_profile = pd.DataFrame({"user_id": np.arange(n_users)})
    item_profile = pd.DataFrame({"item_id": np.arange(n_items)})

    df_train, _ = generate_seq_feature_match(data, "user_id", "item_id", "time", mode=0, neg_ratio=0)
    x_train = gen_model_input(df_train, user_profile, "user_id", item_profile, "item_id", seq_max_len=8)
    # labels are unused in in-batch mode; keep zero array for shape alignment
    y_train = np.zeros(len(df_train))

    user_features = [
        SparseFeature("user_id",
                      n_users,
                      embed_dim=8),
        SequenceFeature("hist_item_id",
                        n_items,
                        embed_dim=8,
                        pooling="mean",
                        shared_with="item_id"),
    ]
    item_features = [SparseFeature("item_id", n_items, embed_dim=8)]

    dg = MatchDataGenerator(x_train, y_train)
    train_dl, _, _ = dg.generate_dataloader(x_train, df_to_dict(item_profile), batch_size=8, num_workers=0)

    model = DSSM(user_features, item_features, user_params={"dims": [16]}, item_params={"dims": [16]})
    return train_dl, model


def test_match_trainer_inbatch_flow_runs_and_updates():
    train_dl, model = _build_small_match_dataloader()

    trainer = MatchTrainer(model, mode=0, in_batch_neg=True, in_batch_neg_ratio=3, sampler_seed=2, n_epoch=1, device="cpu")
    trainer.train_one_epoch(train_dl, log_interval=100)

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


@pytest.mark.parametrize("mode, temperature", [(0, 0.05), (2, 1.0)])
@pytest.mark.parametrize("repeated_items", [False, True])
def test_full_inbatch_matches_reference_and_updates_both_towers(monkeypatch, mode, temperature, repeated_items):
    train_dl, model = _build_small_match_dataloader()
    model.temperature = temperature
    x, y = next(iter(train_dl))
    x['item_id'] = torch.arange(len(y))
    if repeated_items:
        x['item_id'][1] = x['item_id'][0]
    trainer = MatchTrainer(model, mode=mode, in_batch_neg=True, optimizer_fn=torch.optim.SGD, optimizer_params={"lr": 0.01})

    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("Full-batch training must not sample or gather negative indices")

    monkeypatch.setattr("torch_rechub.trainers.match_trainer.inbatch_negative_sampling", unexpected_sampling)
    monkeypatch.setattr("torch_rechub.trainers.match_trainer.gather_inbatch_logits", unexpected_sampling)
    with torch.no_grad():
        user_emb = F.normalize(model.user_tower(x), dim=-1)
        item_emb = F.normalize(model.item_tower(x), dim=-1)
        logits = user_emb @ item_emb.T / temperature
        same_item = x['item_id'][:, None] == x['item_id'][None, :]
        same_item.fill_diagonal_(False)
        logits = logits.masked_fill(same_item, float('-inf'))
        expected_loss = F.cross_entropy(logits, torch.arange(len(y))).item()
    user_before = [p.detach().clone() for p in model.user_mlp.parameters()]
    item_before = [p.detach().clone() for p in model.item_mlp.parameters()]
    singleton = ({key: value[:1] for key, value in x.items()}, y[:1])

    with pytest.warns(RuntimeWarning, match="Skipped 1"):
        actual_loss = trainer.train_one_epoch([(x, y), singleton])

    assert actual_loss == pytest.approx(expected_loss, rel=1e-5)
    assert any(not torch.equal(before, after) for before, after in zip(user_before, model.user_mlp.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(item_before, model.item_mlp.parameters()))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_inbatch_rejects_an_epoch_without_negatives():
    train_dl, model = _build_small_match_dataloader()
    x, y = next(iter(train_dl))
    trainer = MatchTrainer(model, in_batch_neg=True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    singleton = ({key: value[:1] for key, value in x.items()}, y[:1])

    with pytest.raises(ValueError, match="No usable in-batch"):
        trainer.train_one_epoch([singleton])
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    x['item_id'] = torch.ones_like(x['item_id'])
    with pytest.raises(ValueError, match="two different item IDs"):
        trainer.train_one_epoch([(x, y)])
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_full_inbatch_validates_temperature_and_single_device():
    _, model = _build_small_match_dataloader()
    for temperature in (0.0, -0.1, float("nan"), float("inf")):
        model.temperature = temperature
        with pytest.raises(ValueError, match="temperature"):
            MatchTrainer(model, in_batch_neg=True)

    model.temperature = 1.0
    with pytest.raises(ValueError, match="single device"):
        MatchTrainer(model, in_batch_neg=True, gpus=[0, 1])

    # Point-wise DSSM still uses its original BCE path without temperature scaling.
    model.temperature = 0.0
    trainer = MatchTrainer(model)
    assert isinstance(trainer.criterion, torch.nn.BCELoss)


@pytest.mark.parametrize("hard_negative", [False, True])
def test_inbatch_sampling_excludes_same_item(hard_negative):
    item_ids = torch.tensor([7, 7, 8, 9])
    scores = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    options = {"neg_ratio": 3, "hard_negative": hard_negative, "item_ids": item_ids}
    sampled = inbatch_negative_sampling(scores, generator=torch.Generator().manual_seed(2), **options)
    assert sampled.shape == (4, 2)  # Two valid candidates for each of the repeated items.
    assert (item_ids[sampled] != item_ids[:, None]).all()
    assert torch.equal(sampled, inbatch_negative_sampling(scores, generator=torch.Generator().manual_seed(2), **options))
    assert inbatch_negative_sampling(scores, item_ids=torch.ones(4)).shape == (4, 0)


@pytest.mark.parametrize("mode, hard_negative", [(0, False), (0, True), (1, False), (1, True), (2, False)])
def test_sampled_inbatch_with_repeated_items_has_finite_gradients(mode, hard_negative):
    train_dl, model = _build_small_match_dataloader()
    x, y = next(iter(train_dl))
    x['item_id'] = torch.tensor([7, 7, 8, 9, 7, 8, 9, 9])
    trainer = MatchTrainer(model, mode=mode, in_batch_neg=True, in_batch_neg_ratio=3, hard_negative=hard_negative, sampler_seed=2)
    assert np.isfinite(trainer.train_one_epoch([(x, y)]))
    assert model.mode is None
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("ratio", [1, 3, None])
@pytest.mark.parametrize("item_ids", [[7, 7, 8, 9], [7, 7, 7, 8]])
def test_ranking_inbatch_excludes_same_item_and_keeps_fields_together(ratio, item_ids):
    trainer = CTRTrainer(torch.nn.Linear(1, 1), in_batch_neg=True, item_features=['cate', 'item'], item_id_feature='item', in_batch_neg_ratio=ratio, sampler_seed=2)
    x = {'user': torch.arange(4), 'item': torch.tensor(item_ids), 'cate': torch.tensor(item_ids) * 10, 'history': torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])}
    paired, labels = trainer._inbatch_data(x, torch.ones(4))
    owners = paired['user']
    negative = labels == 0
    assert (paired['item'][negative] != x['item'][owners[negative]]).all()
    assert torch.equal(paired['cate'], paired['item'] * 10)
    assert torch.equal(paired['history'], x['history'][owners])
    assert int((labels == 1).sum()) == 4
    negatives = int((x['item'][:, None] != x['item'][None, :]).sum()) if ratio is None else 4 * ratio
    assert len(labels) == 4 + negatives
    for owner in range(4):
        row_labels = labels[owners == owner]
        assert row_labels[0] == 1
        assert (row_labels[1:] == 0).all()
    if ratio is not None:
        trainer._sampler_generator.manual_seed(2)
        repeated, repeated_labels = trainer._inbatch_data(x, torch.ones(4))
        assert torch.equal(paired['item'], repeated['item'])
        assert torch.equal(labels, repeated_labels)
    x['item'][:] = 7
    assert trainer._inbatch_data(x, torch.ones(4)) is None


def test_inbatch_item_id_configuration_and_shape():
    train_dl, model = _build_small_match_dataloader()
    x, _ = next(iter(train_dl))
    trainer = MatchTrainer(model, in_batch_neg=True)
    assert torch.equal(trainer._inbatch_item_ids(x), x['item_id'])
    x['item_id'] = x['item_id'].unsqueeze(1)
    assert trainer._inbatch_item_ids(x).shape == (8,)
    x['item_id'] = x['item_id'].expand(-1, 2)
    with pytest.raises(ValueError, match="one item ID"):
        trainer._inbatch_item_ids(x)
    with pytest.raises(ValueError, match="included in item_features"):
        CTRTrainer(torch.nn.Linear(1, 1), in_batch_neg=True, item_features=['cate'], item_id_feature='item')


def test_default_item_id_is_independent_of_feature_order():
    x = {'user_id': torch.arange(4), 'cate_id': torch.ones(4, dtype=torch.long), 'item_id': torch.tensor([7, 7, 8, 9])}
    y = torch.ones(4)
    ranking = CTRTrainer(torch.nn.Linear(1, 1), in_batch_neg=True, item_features=['cate_id', 'item_id'])
    paired, labels = ranking._inbatch_data(x, y)
    negative = labels == 0
    assert (paired['item_id'][negative] != x['item_id'][paired['user_id'][negative]]).all()
    model = DSSM([SparseFeature('user_id', 4, 8)], [SparseFeature('cate_id', 2, 8), SparseFeature('item_id', 16, 8)], user_params={'dims': [8]}, item_params={'dims': [8]})
    matching = MatchTrainer(model, in_batch_neg=True)
    assert matching.item_id_feature == ranking.item_id_feature == 'item_id'
    assert torch.equal(matching._inbatch_item_ids(x), x['item_id'])
    assert np.isfinite(matching.train_one_epoch([(x, y)]))
    renamed = {('sku' if name == 'item_id' else name): values for name, values in x.items()}
    with pytest.raises(ValueError, match="Missing item ID column: item_id"):
        matching._inbatch_item_ids(renamed)
    matching.item_id_feature = 'sku'
    assert torch.equal(matching._inbatch_item_ids(renamed), x['item_id'])


def test_custom_towers_need_no_feature_metadata():

    class CustomTowers(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.mode = None
            self.embedding = torch.nn.Embedding(16, 4)

        def user_tower(self, x):
            return self.embedding(x['user_id'])

        def item_tower(self, x):
            return self.embedding(x['item_id'])

        def forward(self, x):
            return self.user_tower(x), self.item_tower(x)

    model = CustomTowers()
    trainer = MatchTrainer(model, in_batch_neg=True)
    x = {'user_id': torch.arange(3), 'item_id': torch.tensor([7, 7, 8])}
    assert np.isfinite(trainer.train_one_epoch([(x, torch.ones(3))]))
    assert model.mode is None


def test_sasrec_default_positive_sequence_ids():
    features = [SequenceFeature(name, 16, 4, pooling='concat') for name in ('seq', 'pos', 'neg')]
    model = SASRec(features, max_len=3, dropout_rate=0, num_blocks=1, num_heads=1)
    trainer = MatchTrainer(model, in_batch_neg=True)
    x = {'seq': torch.tensor([[0, 1, 2], [3, 4, 0]]), 'pos': torch.tensor([[0, 2, 7], [4, 8, 0]]), 'item_id': torch.tensor([1, 1])}
    # The sequence model's actual positives win over an unrelated item_id column.
    assert torch.equal(trainer._inbatch_item_ids(x), torch.tensor([7, 8]))
    model.mode = 'in_batch'
    torch.testing.assert_close(model.item_tower(x).squeeze(1), model.item_emb.embed_dict['seq'](model.get_item_ids(x)))
    model.mode = None
    assert np.isfinite(trainer.train_one_epoch([(x, torch.zeros(2))]))


@pytest.mark.parametrize('item_column', ['movie_id', 'target_item_id'])
def test_existing_item_column_names_need_no_extra_argument(item_column):
    x = {'user_id': torch.arange(3), 'cate_id': torch.ones(3, dtype=torch.long), item_column: torch.tensor([7, 7, 8])}
    ranking = CTRTrainer(torch.nn.Linear(1, 1), in_batch_neg=True, item_features=['cate_id', item_column])
    paired, labels = ranking._inbatch_data(x, torch.ones(3))
    negative = labels == 0
    assert (paired[item_column][negative] != x[item_column][paired['user_id'][negative]]).all()
    model = DSSM([SparseFeature('user_id', 3, 4)], [SparseFeature('cate_id', 2, 4), SparseFeature(item_column, 16, 4)], user_params={'dims': [4]}, item_params={'dims': [4]})
    matching = MatchTrainer(model, in_batch_neg=True)
    assert torch.equal(matching._inbatch_item_ids(x), x[item_column])
