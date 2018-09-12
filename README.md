# 中国法研杯比赛

## 法律数据集

### 文件组成
**cail2018_big.json**: 171w

### 数据组成
数据中涉及 **183个法条**、**202个罪名**，均为刑事案件

### 数据清洗
数据中筛除了刑法中前101条(前101条并不涉及罪名)，并且为了方便进行模型训练，将罪名和法条数量少于30的类删去。

### 数据格式
数据利用json格式储存，每一行为一条数据，每条数据均为一个字典
##### 字段及意义
* **fact**: 事实描述
* **meta**: 标注信息，标注信息中包括:
	* **criminals**: 被告(数据中均只含一个被告)
	* **punish\_of\_money**: 罚款(单位：元)
	* **accusation**: 罪名
	* **relevant\_articles**: 相关法条
	* **term\_of\_imprisonment**: 刑期
		刑期格式(单位：月)
		* **death\_penalty**: 是否死刑
		* **life\_imprisonment**: 是否无期
		* **imprisonment**: 有期徒刑刑期

## 数据处理
停用词
地名，人名，一般停用词。

分词
Python包：jieba。

## 模型
此部分涉及两个模型：TextCNN，Attention。

##  代码框架
|- ckpt       # 保存训练好的模型<br/>
|- data　　　    # 预处理得到的数据<br/>
|- data_raw　　　　   # 原始数据<br/>
|- log                    # 训练日志<br/>
|- models　　　　　  # 模型代码<br/>
|　　|- Attention_TextCNN               # 模型名称<br/>
|　　|　　|- network.py　　　   　　    # 定义网络结构<br/>
|　　|　　|- train.py　　　　  　　      # 模型训练<br/>
|　　|　　|- predict.py　　　  　　    　# 模型预测<br/>
|- process_data　　　　　  　　　　　  # 预处理<br/>
|- scores　　　　   　　　　  　　　   # 预测的结果<br/>
|- summary　　　　　　　　  　       # tensorboard数据<br/>
|- data_helper.py　　　　　   　        # 数据处理辅助函数<br/>
|- evaluator.py　                       # 评价函数<br/>
|- utils.py                             # 其他函数<br/>

下面是我实验中的一些环境依赖，版本只提供参考。

|环境/库|版本|
|:---------:|----------|
|Ubuntu|16.04 LTS|
|python|3.5.0|
|tensorflow-gpu|1.4.0|

## 代码运行
law_id.py --> embed2ndarray.py --> fact2dic_law2id.py --> fact2words.py --> word2id.py --> batch_data.py

train.py --> predict.py

## 结果
任务一:
42	shelley	86.91	85.34	85.81

任务二:
41	shelley	84.63	82.87	83.40

## 参考文献
（1）TextCNN：
Kim Y. Convolutional Neural Networks for Sentence Classification[J]. Eprint Arxiv, 2014.<br/>
Conneau A, Schwenk H, Barrault L, et al. Very Deep Convolutional Networks for Text Classification[J]. 2017:1107-1116.
Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[J]. 2014:1-9.<br/>

（2）Attention：
Yang Z, Yang D, Dyer C, et al. Hierarchical Attention Networks for Document Classification[C]// Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2017:1480-1489.<br/>

