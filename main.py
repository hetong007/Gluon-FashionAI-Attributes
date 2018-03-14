import mxnet as mx
import numpy as np
import os, time, logging, math

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

task_list = {
    'collar_design_labels': 5,
    'skirt_length_labels': 6,
    'lapel_design_labels': 5,
    'neckline_design_labels': 10,
    'coat_length_labels': 8,
    'neck_design_labels': 5,
    'pant_length_labels': 6,
    'sleeve_length_labels': 9}

momentum = 0.9
wd = 1e-4
epochs = 40

batch_size = 64
lr = 1e-3
num_gpu = 4
ctx = [mx.gpu(i) for i in range(num_gpu)]
batch_size = batch_size*num_gpu

logging.basicConfig(level=logging.INFO,
                    handlers = [
                        logging.StreamHandler(),
                        logging.FileHandler('training.log')
                    ])

def get_ap(labels, outputs):
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

def transform_val_normal(data, label):
    im = data.astype('float32') / 255
    im = image.resize_short(im, 256)
    im, _ = image.center_crop(im, (224, 224))
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return (im, nd.array([label]).asscalar())

def transform_val_tencrop(im, label):
    im = im.astype('float32') / 255
    im = image.resize_short(im, 256)
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    im = ten_crop(im, (224, 224))
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

def test_normal(net, val_data, ctx):
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
        ap, cnt = get_ap(label, outputs)
        AP += ap
        AP_cnt += cnt
    _, val_acc = metric.get()
    return ((val_acc, AP / AP_cnt, val_loss / len(val_data)))

def test_tencrop(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    AP = 0.
    AP_cnt = 0
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = []
        for d in data:
            n = d.shape[0]
            outs = []
            for i in range(n):
                out = net(d[i])
                out = nd.SoftmaxActivation(out).mean(axis=0)
                outs.append(out.asnumpy().tolist())
            outputs.append(nd.array(outs))
        metric.update(label, outputs)
        ap, cnt = get_ap(label, outputs)
        AP += ap
        AP_cnt += cnt
    _, val_acc = metric.get()
    return ((val_acc, AP / AP_cnt))

def train(task, task_num_class):
    logging.info('Start Training for Task: %s\n' % (task))

    # Initialize the net with pretrained model
    pretrained_net = models.resnet50_v2(pretrained=True)

    finetune_net = models.resnet50_v2(classes=task_num_class)
    model_name = 'resnet50_v2'
    finetune_net.features = pretrained_net.features
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    # for v in finetune_net.collect_params().values():
    #     if 'dense' in v.name:
    #         setattr(v, 'lr_mult', 10)
    finetune_net.hybridize()

    # Define DataLoader
    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join('train_valid', task, 'train'),
            transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=32, last_batch='discard')

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join('train_valid', task, 'val'),
            transform=transform_val_normal),
        batch_size=batch_size, shuffle=False, num_workers = 32)

    # Define Trainer
    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
        'learning_rate': lr, 'momentum': momentum,'wd': wd})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    iteration = 0
    
    # Start Training
    for epoch in range(epochs):
        if (epoch+1) % 10 == 0:
            trainer.set_learning_rate(trainer.learning_rate()*0.5)
        tic = time.time()
        train_loss = 0
        num_batch = len(train_data)
        metric.reset()

        # if epoch == 40:
        #     trainer.set_learning_rate(lr*0.1)
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
            ap, cnt = get_ap(label, outputs)
            AP += ap
            AP_cnt += cnt

            iteration += 1
            progressbar(i, num_batch-1)

        train_map = AP / AP_cnt
        _, train_acc = metric.get()
        train_loss /= num_batch
        if val_data is None:
            logging.info('[Epoch %d] train-acc: %.3f, train-map: %.3f, train-loss: %.3f, time: %.1f' %
                (epoch, train_acc, train_map, train_loss, time.time() - tic))
        else:
            # val_acc, val_map= test(finetune_net, val_data, ctx)
            val_acc, val_map, val_loss = test_normal(finetune_net, val_data, ctx)
            logging.info('[Epoch %d] Train-acc: %.3f, mAP: %.3f, loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1f' %
                 (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time() - tic))

    logging.info('\n')
    return (finetune_net)

if __name__ == '__main__':
    net_dict = {}
    for task, task_num_class in task_list.items():
        net_dict[task] = train(task, task_num_class)

    logging.info('Training Finished. Starting Validation.\n')
    # Validate All Network is Working

    for task in task_list.keys():
        val_data = gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(
                os.path.join('train_valid', task, 'val'),
                transform=transform_val_normal),
            batch_size=batch_size, shuffle=False, num_workers = 32)
        val_acc, val_map, val_loss = test_normal(net_dict[task], val_data, ctx)
        logging.info('[Validation for %s] Val-acc: %.3f, mAP: %.3f, loss: %.3f' % 
            (task, val_acc, val_map, val_loss))

    logging.info('Validation Finished. Starting Prediction.\n')
    f_out = open('submission.csv', 'w')
    with open('rank/Tests/question.csv', 'r') as f_in:
        lines = f_in.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    n = len(tokens)
    cnt = 0
    for path, task, _ in tokens:
        img_path = os.path.join('rank', path)
        with open(img_path, 'rb') as f:
            img = image.imdecode(f.read())
        data = transform_predict(img)
        out = net_dict[task](data.as_in_context(mx.gpu(0)))
        out = nd.SoftmaxActivation(out).mean(axis=0)

        pred_out = ';'.join([str(o) for o in out.asnumpy().tolist()])
        line_out = ','.join([path, task, pred_out])
        f_out.write(line_out + '\n')
        cnt += 1
        progressbar(cnt, n)
    f_out.close()


