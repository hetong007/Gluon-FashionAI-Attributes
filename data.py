import mxnet
from mxnet import gluon, image
import os, shutil, random

# Read label.csv
# For each task, make folders, and copy picture to corresponding folders

label_dir = 'base/Annotations/label.csv'
warmup_label_dir = 'web/Annotations/skirt_length_labels.csv'

label_dict = {'coat_length_labels': [],
              'lapel_design_labels': [],
              'neckline_design_labels': [],
              'skirt_length_labels': [],
              'collar_design_labels': [],
              'neck_design_labels': [],
              'pant_length_labels': [],
              'sleeve_length_labels': []}

task_list = label_dict.keys()

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

with open(label_dir, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        label_dict[task].append((path, label))

mkdir_if_not_exist(['train_valid'])

for task, path_label in label_dict.items():
    mkdir_if_not_exist(['train_valid',  task])
    train_count = 0
    n = len(path_label)
    m = len(list(path_label[0][1]))

    for mm in range(m):
        mkdir_if_not_exist(['train_valid', task, 'train', str(mm)])
        mkdir_if_not_exist(['train_valid', task, 'val', str(mm)])

    random.shuffle(path_label)
    for path, label in path_label:
        label_index = list(label).index('y')
        src_path = os.path.join('base', path)
        if train_count < n * 0.9:
            shutil.copy(src_path,
                        os.path.join('train_valid', task, 'train', str(label_index)))
        else:
            shutil.copy(src_path,
                        os.path.join('train_valid', task, 'val', str(label_index)))
        train_count += 1

# Add warmup data to skirt task

label_dict = {'skirt_length_labels': []}

with open(warmup_label_dir, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        label_dict[task].append((path, label))

for task, path_label in label_dict.items():
    train_count = 0
    n = len(path_label)
    m = len(list(path_label[0][1]))

    random.shuffle(path_label)
    for path, label in path_label:
        label_index = list(label).index('y')
        src_path = os.path.join('web', path)
        if train_count < n * 0.9:
            shutil.copy(src_path,
                        os.path.join('train_valid', task, 'train', str(label_index)))
        else:
            shutil.copy(src_path,
                        os.path.join('train_valid', task, 'val', str(label_index)))
        train_count += 1

