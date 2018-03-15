# Gluon-FashionAI-Attributes

This is the repo for [MXNet/Gluon](http://mxnet.incubator.apache.org/) benchmark scripts for the [FashionAI](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.505c3a26Oet3cf&raceId=231649) competition by Alibaba.

1. Download and untar the data files into this folder, the structure should look like 
```
Gluon-FashionAI-Attributes
├── data
├── data.py
├── FashionAI-Attributes-Skirt.ipynb
├── main.py
└── README.md
```
2. Execute `python data.py` to prepare the `train_valid` folder for train and validation split.
3. Execute `python main.py` to train and predict for all eight tasks.
4. Submit `submission.csv` via the competition portal.

The result will have mAP around 0.95 and Basic Precision around 0.84 on the board.

