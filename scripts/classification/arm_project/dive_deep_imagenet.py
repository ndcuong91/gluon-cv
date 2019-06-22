import argparse, time

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from mxnet.gluon.data.vision import transforms
from arm_network import get_arm_network

# number of GPUs to use
num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
model_name='3.4.2'

# Get the model ResNet50_v2, with 10 output classes
#net = get_model('ResNet50_v2', classes=1000)
#net.initialize(mx.init.MSRAPrelu(), ctx = ctx)

net = get_arm_network(model_name, ctx)
net.output = nn.Dense(1000)
net.initialize(mx.init.MSRAPrelu(), ctx = ctx)
net.hybridize()

jitter_param = 0.4
lighting_param = 0.1
mean_rgb = [123.68, 116.779, 103.939]
std_rgb = [58.393, 57.12, 57.375]

from gluoncv.data import imagenet
import argparse, time, logging, os, math


batch_size=256
input_size = 112
num_workers=8
crop_ratio=0.875
resize = int(math.ceil(input_size / crop_ratio))
data_dir='~/.mxnet/datasets/imagenet'
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    normalize
])
transform_test = transforms.Compose([
    transforms.Resize(resize, keep_ratio=True),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    normalize
])
train_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
val_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Learning rate decay factor
lr_decay = 0.1
# Epochs where learning rate decays
lr_decay_epoch = [30, 60, 90, np.inf]

# Nesterov accelerated gradient descent
optimizer = 'nag'
# Set parameters
optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}

# Define our trainer for net
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)


loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_history = TrainingHistory(['training-top1-err', 'training-top5-err',
                                 'validation-top1-err', 'validation-top5-err'])

def test(ctx, val_data):
    acc_top1_val = mx.metric.Accuracy()
    acc_top5_val = mx.metric.TopKAccuracy(5)
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        acc_top1_val.update(label, outputs)
        acc_top5_val.update(label, outputs)

    _, top1 = acc_top1_val.get()
    _, top5 = acc_top5_val.get()
    return (1 - top1, 1 - top5)

epochs = 120
lr_decay_count = 0
log_interval = 200
lr_decay_period=0

best_val_score=1
save_dir='output'

for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    acc_top1.reset()
    acc_top5.reset()

    if lr_decay_period == 0 and epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    for i, batch in enumerate(train_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        with ag.record():
            outputs = [net(X) for X in data]
            loss = [loss_fn(yhat, y) for yhat, y in zip(outputs, label)]
        ag.backward(loss)
        trainer.step(batch_size)
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
        if log_interval and not (i + 1) % log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            err_top1, err_top5 = (1-top1, 1-top5)
            print('Epoch[%d] Batch [%d]     Speed: %f samples/sec   top1-err=%f     top5-err=%f'%(
                      epoch, i, batch_size*log_interval/(time.time()-btic), err_top1, err_top5))
            btic = time.time()

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    err_top1, err_top5 = (1-top1, 1-top5)

    err_top1_val, err_top5_val = test(ctx, val_data)
    train_history.update([err_top1, err_top5, err_top1_val, err_top5_val])

    print('[Epoch %d] training: err-top1=%f err-top5=%f'%(epoch, err_top1, err_top5))
    print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
    print('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

    if err_top1_val < best_val_score:
        best_val_score = err_top1_val
        net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params' % (save_dir, best_val_score, model_name, epoch))
        trainer.save_states('%s/%.4f-imagenet-%s-%d-best.states' % (save_dir, best_val_score, model_name, epoch))

train_history.plot(['training-top1-err', 'validation-top1-err'])