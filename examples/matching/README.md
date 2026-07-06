# 召回

## Movielens

使用ml-1m数据集，使用其中原始特征7个user特征`'user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip',"cate_id"`，2个item特征`"movie_id", "cate_id"`，一共9个sparse特征。

- 构造用户观看历史特征``hist_movie_id``，使用`mean`池化该序列embedding
- 使用随机负采样构造负样本 (sample_method=0)，内含随机负采样、word2vec负采样、流行度负采样、Tencent负采样等多种方法
- 将每个用户最后一条观看记录设置为测试集
- 原始数据下载地址：https://grouplens.org/datasets/movielens/1m/
- 处理完整数据csv下载地址：https://cowtransfer.com/s/5a3ab69ebd314e

以下指标使用example中的相同参数在ml-1m上测试得到

| Model\Metrics | Hit@100 | Recall@100 | Precision@100 |
|---------------|---------|------------|---------------|
| DSSM          | 14.74%  | 14.74%     | 0.15%         |
| YoutubeDNN    | 6.27%   | 6.27%      | 0.06%         |
| FacebookDSSM  | 2.85%   | 2.85%      | 0.03%         |
| YoutubeSBC    | 17.83%  | 17.83%     | 0.18%         |



## 双塔模型对比

| 模型         | 学习模式   | 损失函数  | 样本构造                                                     | label                              |
| ------------ | ---------- | --------- | ------------------------------------------------------------ | ---------------------------------- |
| DSSM         | point-wise | BCE       | 全局负采样，一条负样本对应label 0                            | 1或0                               |
| YoutubeDNN   | list-wise  | CE        | 全局负采样，每条正样本对应k条负样本                          | 0（item_list中第一个位置为正样本） |
| YoutubeSBC   | list-wise  | CE        | Batch内随机负采样，每条正样本对应k条负样本，加入采样权重做纠偏处理 | 0（item_list中第一个位置为正样本） |
| FacebookDSSM | pair-wise  | BPR/Hinge | 全局负采样，每条正样本对应1个负样本，需扩充负样本item其他属性特征 | 无label                            |



## YiDian-News

一点资讯 CTR 比赛数据集的处理版，用于点击率预估（CTR Prediction）和推荐系统排序任务。

### 全量数据
* 原始数据为NewsDataset.zip[(下载链接)](https://aistudio.baidu.com/dataset/detail/389517)
主数据按体积分为 32 个 CSV 分卷：

- `yidian_news_processed_part_0001.csv`
- ...
- `yidian_news_processed_part_0032.csv`

调试样例为 `_sample.csv`，包含 1000 行数据。压缩包为 `dataset.zip`。

如需恢复为单个完整 CSV，可在 `YiDian_News_dataset/` 目录运行：

```powershell
python merge_csv_big.py "yidian_news_processed_part_*.csv" yidian_news_processed.csv
```

合并后的 `yidian_news_processed.csv` 可直接用于训练或离线分析。数据体量较大，快速验证流程时建议优先使用 `_sample.csv`。

#### 数据列表

处理后的 CSV 共 22 列：

1. 行为字段：`userId`、`itemId`、`showTime`、`network`、`refresh`、`showPos`、`click`、`duration`；
2. 用户字段：`deviceName`、`OS`、`province`、`city`、`age_0_24`、`age_25_29`、`age_30_39`、`age_40plus`、`female`、`male`；
3. 文章字段：`publishTime`、`imageNum`、`cate1`、`cate2`。

#### 数据项说明

1. 网络环境：0 未知；1 离线；2 WiFi；3 2G；4 3G；5 4G；
2. 刷新次数：用户打开 APP 后推荐页的刷新次数，直到退出 APP 则清零；
3. `click` 为二分类标签，0 表示未点击，1 表示点击；
4. `duration` 为消费时长，单位为秒；
5. 原始训练数据取自用户历史 12 天的行为日志，测试数据采样自第 13 天的用户展现日志。



## Million Song Dataset
百万歌曲音乐数据集（事实上这个数据集很很多种，同时也有不同社区对它进行处理，我们只选取它数据量比较少且文件结构简单的一个）

### 数据集说明
1.这个项目由The Echonest和LABRosa一起完成

2.数据集主要是多年间外国音乐的量化特征，包含了百万用户对几十万首歌曲的播放记录（train_triplets.txt，2.9G）和这些歌曲的详细信息（triplets_metadata.db，700M）。

### 数据格式
1.用户的播放记录数据集train_triplets.txt:`用户,歌曲,播放次数`，其中用户和歌曲都匿名

2.歌曲的详细信息数据集triplets_metadata.db:`歌曲的发布时间,作者,作者热度,...`等

由于数据集很大，测试训练时可以从.txt文件中选取200万条数据作为我们的数据集。


[数据集下载](http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip)

[数据集说明](http://millionsongdataset.com/tasteprofile/)

## Book-Crossing

针对美国 Book-crossing 网站关于用户对书籍的评级行为进行分析，数据包含 27.8万个匿名用户，提供 115万 个评级(显式/隐式)，涉及约 27.1万本书。增删改查合并后的数据包含约71.9万个评级。

[数据集下载和说明](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)


## Session based recommendation datasets
#### 序列推荐的基准数据集目前包括:
* YOOCHOOSE: [ACM RecSys Challenge 2015](https://recsys.acm.org/recsys15/challenge/) 所使用的电商网站点击流数据集。构建序列推荐数据只用到该数据集的 `train-item-views.csv` 文件。该文件包含四个原始特征：`"session_id", "item_id", "time","category"`。该数据目前可以在 [Kaggle](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015) 网站下载。
* DIGINETICA: [CIKM Cup 2016](https://competitions.codalab.org/competitions/11161) 使用的电商网站点击流数据集。构建序列推荐数据只用到该数据集的 `yoochoose-clicks.dat` 文件。该文件包含五个原始特征：`"sessionId", “userId”, "itemId", "time", "timeframe", "eventdate"`。该数据目前可以在 [google drive](https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c?resourcekey=0-3k4O5YlwnZf0cNeTZ5Y_Uw) 下载。

#### 测试结果

|       | YOOC.1/64<br> Recall@20 | YOOC.1/64<br> Recall@20 | YOOC.1/64<br> Recall@20 | YOOC.1/64<br> Recall@20 | DIGI.<br> Recall@20 | DIGI.<br> Recall@20 |
|:-----:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-------------------:|:-------------------:|
|  NARM |          0.6746         |          0.2827         |          0.7028         |          0.2909         |        0.5829       |        0.2603       |
| STAMP |          0.6675         |          0.2859         |          0.7079         |          0.3074         |        0.5578       |        0.2303       |

* __Neural Attentive Session-based Recommendation__ ([Li et al., CIKM'17](https://arxiv.org/abs/1711.04725.
* __STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation__  ([Liu et al., KDD'18](https://dl.acm.org/doi/10.1145/3219819.3219950))
* 注：以上指标可使用论文中实验章节提到的训练参数测试得到。排序指标 `top_k` 则需要调整成 `20`。
