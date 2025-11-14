"""
简单的测试脚本，验证embedding正则化功能是否正确工作
"""
import torch
import torch.nn as nn
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.basic.layers import EmbeddingLayer


def test_embedding_regularization():
    """测试EmbeddingLayer的正则化损失计算"""
    print("=" * 60)
    print("测试1: EmbeddingLayer正则化损失计算")
    print("=" * 60)
    
    # 创建特征，指定L1和L2正则化参数
    features = [
        SparseFeature("user_id", vocab_size=100, embed_dim=8, l1_reg=0.001, l2_reg=0.0001),
        SparseFeature("item_id", vocab_size=200, embed_dim=8, l1_reg=0.002),
        SparseFeature("category", vocab_size=50, embed_dim=8),  # 无正则化
    ]
    
    # 创建EmbeddingLayer
    embedding_layer = EmbeddingLayer(features)
    
    # 检查reg_dict是否正确存储
    print(f"reg_dict: {embedding_layer.reg_dict}")
    assert "user_id" in embedding_layer.reg_dict
    assert "item_id" in embedding_layer.reg_dict
    assert "category" in embedding_layer.reg_dict
    
    # 获取正则化损失
    reg_loss = embedding_layer.get_regularization_loss()
    print(f"正则化损失: {reg_loss}")
    assert reg_loss > 0, "正则化损失应该大于0"
    
    print("✅ 测试1通过\n")


def test_feature_parameters():
    """测试特征类中的L1/L2参数"""
    print("=" * 60)
    print("测试2: 特征类中的L1/L2参数")
    print("=" * 60)
    
    # 测试SparseFeature
    sparse_feat = SparseFeature("test", vocab_size=100, embed_dim=8, l1_reg=0.001, l2_reg=0.0001)
    print(f"SparseFeature - l1_reg: {sparse_feat.l1_reg}, l2_reg: {sparse_feat.l2_reg}")
    assert sparse_feat.l1_reg == 0.001
    assert sparse_feat.l2_reg == 0.0001
    
    # 测试SequenceFeature
    seq_feat = SequenceFeature("hist", vocab_size=100, embed_dim=8, l1_reg=0.002)
    print(f"SequenceFeature - l1_reg: {seq_feat.l1_reg}, l2_reg: {seq_feat.l2_reg}")
    assert seq_feat.l1_reg == 0.002
    assert seq_feat.l2_reg == 0.0
    
    # 测试默认值（无正则化）
    default_feat = SparseFeature("default", vocab_size=100, embed_dim=8)
    print(f"Default - l1_reg: {default_feat.l1_reg}, l2_reg: {default_feat.l2_reg}")
    assert default_feat.l1_reg == 0.0
    assert default_feat.l2_reg == 0.0
    
    print("✅ 测试2通过\n")


def test_backward_compatibility():
    """测试向后兼容性"""
    print("=" * 60)
    print("测试3: 向后兼容性")
    print("=" * 60)
    
    # 创建不指定正则化参数的特征
    features = [
        SparseFeature("user_id", vocab_size=100, embed_dim=8),
        SparseFeature("item_id", vocab_size=200, embed_dim=8),
    ]
    
    embedding_layer = EmbeddingLayer(features)
    reg_loss = embedding_layer.get_regularization_loss()
    
    print(f"无正则化参数时的损失: {reg_loss}")
    assert reg_loss == 0.0, "无正则化参数时损失应该为0"
    
    print("✅ 测试3通过\n")


if __name__ == "__main__":
    print("\n开始测试embedding正则化功能...\n")
    
    try:
        test_feature_parameters()
        test_embedding_regularization()
        test_backward_compatibility()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

