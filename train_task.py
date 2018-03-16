import mxnet as mx
import numpy as np
import os, time, logging, math, argparse

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

def parse_args():
    parser = argparse.ArgumentParser(description='Gluon for FashionAI Competition',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', required=True, type=str,
                        help='name of the classification task')
    parser.add_argument('--model', required=True, type=str,
                        help='name of the pretrained model from model zoo.')
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-gpus', default=0, type=int,
                        help='number of gpus to use, 0 indicates cpu only')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-factor', default=0.75, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--lr-steps', default='10,20,30', type=str,
                        help='list of learning rate decay epochs as in str')
    args = parser.parse_args()
    return args

def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for label, output in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int),
                          output.asnumpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = int(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return ((ap, cnt))

def ten_crop(img, size):
    H, W = size
    iH, iW = img.shape[1:3]

    if iH < H or iW < W:
        raise ValueError('image size is smaller than crop size')

    img_flip = img[:, :, ::-1]
    crops = nd.stack(
        img[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img[:, 0:H, 0:W],
        img[:, iH - H:iH, 0:W],
        img[:, 0:H, iW - W:iW],
        img[:, iH - H:iH, iW - W:iW],

        img_flip[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img_flip[:, 0:H, 0:W],
        img_flip[:, iH - H:iH, 0:W],
        img_flip[:, 0:H, iW - W:iW],
        img_flip[:, iH - H:iH, iW - W:iW],
    )
    return (crops)

def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    rand_crop=True, rand_mirror=True,
                                    mean = np.array([0.485, 0.456, 0.406]),
                                    std = np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar())

def transform_val(data, label):
    im = data.astype('float32') / 255
    im = image.resize_short(im, 256)
    im, _ = image.center_crop(im, (224, 224))
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return (im, nd.array([label]).asscalar())

def transform_predict(im):
    im = im.astype('float32') / 255
    im = image.resize_short(im, 256)
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    im = ten_crop(im, (224, 224))
    return (im)

def progressbar(i, n, bar_len=40):
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[%s] %s%s' % (prog_bar, percents, '%'), end = '\r')

def validate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    AP = 0.
    AP_cnt = 0
    val_loss = 0
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
        loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
        ap, cnt = calculate_ap(label, outputs)
        AP += ap
        AP_cnt += cnt
    _, val_acc = metric.get()
    return ((val_acc, AP / AP_cnt, val_loss / len(val_data)))

def train():
    logging.info('Start Training for Task: %s\n' % (task))

    # Initialize the net with pretrained model
    pretrained_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True)

    finetune_net = gluon.model_zoo.vision.get_model(model_name, classes=task_num_class)
    finetune_net.features = pretrained_net.features
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    # Define DataLoader
    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join('data/train_valid', task, 'train'),
            transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='discard')

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join('data/train_valid', task, 'val'),
            transform=transform_val),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    # Define Trainer
    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    lr_counter = 0
    num_batch = len(train_data)

    # Start Training
    for epoch in range(epochs):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*lr_factor)
            lr_counter += 1

        tic = time.time()
        train_loss = 0
        metric.reset()
        AP = 0.
        AP_cnt = 0

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            metric.update(label, outputs)
            ap, cnt = calculate_ap(label, outputs)
            AP += ap
            AP_cnt += cnt

            progressbar(i, num_batch-1)

        train_map = AP / AP_cnt
        _, train_acc = metric.get()
        train_loss /= num_batch

        val_acc, val_map, val_loss = validate(finetune_net, val_data, ctx)

        logging.info('[Epoch %d] Train-acc: %.3f, mAP: %.3f, loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1f' %
                 (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time() - tic))

    logging.info('\n')
    return (finetune_net)

def predict(task):
    logging.info('Training Finished. Starting Prediction.\n')
    f_out = open('submission/%s.csv'%(task), 'w')
    with open('data/rank/Tests/question.csv', 'r') as f_in:
        lines = f_in.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    task_tokens = [t for t in tokens if t[1] == task]
    n = len(task_tokens)
    cnt = 0
    for path, task, _ in task_tokens:
        img_path = os.path.join('data/rank', path)
        with open(img_path, 'rb') as f:
            img = image.imdecode(f.read())
        data = transform_predict(img)
        out = net(data.as_in_context(mx.gpu(0)))
        out = nd.SoftmaxActivation(out).mean(axis=0)

        pred_out = ';'.join(["%.8f"%(o) for o in out.asnumpy().tolist()])
        line_out = ','.join([path, task, pred_out])
        f_out.write(line_out + '\n')
        cnt += 1
        progressbar(cnt, n)
    f_out.close()

# Preparation
args = parse_args()

task_list = {
    'collar_design_labels': 5,
    'skirt_length_labels': 6,
    'lapel_design_labels': 5,
    'neckline_design_labels': 10,
    'coat_length_labels': 8,
    'neck_design_labels': 5,
    'pant_length_labels': 6,
    'sleeve_length_labels': 9
}
task = args.task
task_num_class = task_list[task]

model_name = args.model

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
momentum = args.momentum
wd = args.wd

lr_factor = args.lr_factor
lr_steps = [int(s) for s in args.lr_steps.split(',')] + [np.inf]

num_gpus = args.num_gpus
num_workers = args.num_workers
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = batch_size * max(num_gpus, 1)

logging.basicConfig(level=logging.INFO,
                    handlers = [
                        logging.StreamHandler(),
                        logging.FileHandler('training.log')
                    ])

if __name__ == "__main__":
    net = train()
    predict(task)

