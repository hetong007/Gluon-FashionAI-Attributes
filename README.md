# Gluon-FashionAI-Attributes

This is the repo for [MXNet/Gluon](http://mxnet.incubator.apache.org/) benchmark scripts for the [FashionAI](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.505c3a26Oet3cf&raceId=231649) competition by Alibaba.

The generated submission will have mAP around 0.95 and Basic Precision around 0.84 on the board.

1. Download and untar the data files into `data/` folder, the structure should look like 
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
2. Execute `bash benchmark.sh` to prepare data, train and predict for all eight tasks.
3. Compress and submit `submission/submission.csv` via the competition portal.

The script was tested on a [p3.8xlarge](https://aws.amazon.com/ec2/instance-types/p3/) EC2 instance from AWS. It costs around two and half hours.

