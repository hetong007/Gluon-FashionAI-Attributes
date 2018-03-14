# Gluon-FashionAI-Attributes

This is the repo for tutorial scripts for the [FashionAI](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.505c3a26Oet3cf&raceId=231649) competition by Alibaba.

1. Download and untar the data files into this folder, the structure should be 
```
Gluon-FashionAI-Attributes
├── base
├── data.py
├── main.py
├── rank
├── README.md
└── web
```
2. Execute `python data.py` to prepare the `train_valid` folder for train and validation split.
3. Execute `python main.py` to train and predict for all eight tasks.
4. Submit `submission.csv` via the competition portal.

