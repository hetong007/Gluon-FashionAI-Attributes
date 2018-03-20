# Gluon-FashionAI-Attributes

这是为[阿里天池竞赛——服饰属性标签识别](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.505c3a26Oet3cf&raceId=231649)提供的`gluon`教程与benchmark代码。

[`gluon`教程](FashionAI-Attributes-Skirt.ipynb)会从零开始一步步讲解，带你上手此次竞赛，同时也能帮助理解benchmark代码。

当前Benchmark代码
- 在初赛数据集上能达到大约**0.95**的mAP与**0.84**的准确率，同时还有很大的提升空间。
- 在4块Tesla V100以及32核CPU的 p3.8xlarge 机器上的总运行时间约为2.5小时。

重现benchmark步骤：

1. 前往比赛官网，登录并注册参赛后可在左边“赛题与数据”标签内下载数据。之后将数据解压到`data/`文件夹中。解压后的目录结构应该如下所示：
```
Gluon-FashionAI-Attributes
├── benchmark.sh
├── data
│   ├── base
│   ├── rank
│   └── web
├── FashionAI-Attributes-Skirt.ipynb
├── prepare_data.py
├── README.md
└── train_task.py
```
2. 根据运行环境设置变量
  - `num_gpus`，即GPU的个数，设置为0即为只用CPU训练。
  - `num_workers`，即用来处理数据的进程个数，建议设置为CPU核心个数。
3. 运行`bash benchmark.sh`，这个脚本会自动准备数据，针对每个任务训练模型并预测，以及最后的合并预测。
4. 运行结束后，将`submission/submission.csv`压缩成`zip`格式并通过官网左侧的“提交结果”标签页提交。
