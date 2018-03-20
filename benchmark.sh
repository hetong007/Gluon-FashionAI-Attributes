num_gpus=4
num_workers=32

python prepare_data.py

python train_task.py --task collar_design_labels --model resnet50_v2 --num-gpus $num_gpus -j $num_workers --epochs 40
python train_task.py --task neckline_design_labels --model resnet50_v2 --num-gpus $num_gpus -j $num_workers --epochs 40
python train_task.py --task skirt_length_labels --model resnet50_v2 --num-gpus $num_gpus -j $num_workers --epochs 40
python train_task.py --task sleeve_length_labels --model resnet50_v2 --num-gpus $num_gpus -j $num_workers --epochs 40
python train_task.py --task neck_design_labels --model resnet50_v2 --num-gpus $num_gpus -j $num_workers --epochs 40
python train_task.py --task coat_length_labels --model resnet50_v2 --num-gpus $num_gpus -j $num_workers --epochs 40
python train_task.py --task lapel_design_labels --model resnet50_v2 --num-gpus $num_gpus -j $num_workers --epochs 40
python train_task.py --task pant_length_labels --model resnet50_v2 --num-gpus $num_gpus -j $num_workers --epochs 40

cd submission
cat collar_design_labels.csv neckline_design_labels.csv skirt_length_labels.csv sleeve_length_labels.csv neck_design_labels.csv coat_length_labels.csv lapel_design_labels.csv pant_length_labels.csv > submission.csv

