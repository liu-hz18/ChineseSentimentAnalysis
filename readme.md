# 程序运行说明

##### 2018011446  计84  刘泓尊

liu-hz18@mails.tsinghua.edu.cn

提交的包中已经包含了预处理后的文件，可以直接训练。

## 预处理

如果您需要从头开始预处理数据(非必需)， 您需要下载`Word2Vec`词向量文件`sgns.sogounews.bigram-char`，您可以从我的云盘

https://cloud.tsinghua.edu.cn/f/cb435184e9ed41979d49/

中下载，放入`/word2vec/`文件夹下之后开始预处理过程。

之后您可以在命令行输入

```bash
python main.py pre
```

来进行预处理工作。

## 训练

命令行中输入**一种**模型名称即可开始训练。

```bash
python main.py mlp|cnn|rnn|textcnn|rcnn|bert (选一个)
```

可以训练相应的神经网络. `main.py` 中针对不同网络设置了对应模型的参数，您可以根据需要修改相关参数.

## 输出文件

均保存在`/save/...` 中对应的文件夹下. 

**输出**包括：

参数配置`config.json`、训练日志`*.log`、模型参数`*.pkl`、

模型可视化结构`*.pdf`、使用`TensorBoard` 可视化的`Scalar` 曲线`events.out.tfevents.xxx`等文件。

## 可视化

您可以在最上层文件夹下执行命令

```
tensorboard ‐‐logdir save/...
```

来查看`acc, loss, F1, corr` 等`scalar` 变化的曲线.

特别地，如果您想查看所有模型的可视化结果，请执行命令

```
tensorboard ‐‐logdir save/all/run
```

之后在http://localhost:6006/ 中查看结果。

------

谢谢老师和助教！