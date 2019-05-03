import mxnet as mx
import numpy as np
import os, time, logging, argparse, shutil

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs, TrainingHistory, viz
from gluoncv.model_zoo import get_model
from datetime import datetime

data_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal'
model='resnet152_v2'
classes = 103
input_sz=224
num_training_samples=79640
batch_size=16
epochs=500
log_interval=200
dataset='ZaloAILandmark'
train_path = os.path.join(data_dir, 'train')
test_path = os.path.join(data_dir, 'val')

def parse_opts():
    parser = argparse.ArgumentParser(description='Transfer learning on zaloAIchallenge dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str,default=model,
                        help='name of the pretrained model from model zoo.')
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-gpus', default=1, type=int,
                        help='number of gpus to use, 0 indicates cpu only')
    parser.add_argument('--num_class', default=classes, type=int,
                        help='number of class')
    parser.add_argument('--num_epochs', type=int, default=epochs,
                        help='number of training epochs.')
    parser.add_argument('--input_sz', default=input_sz, type=int,
                        help='resolution of input size')
    parser.add_argument('-b', '--batch-size', default=batch_size, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-factor', default=0.75, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--log-interval', default=log_interval, type=int,
                        help='learning rate decay ratio')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    opts = parser.parse_args()
    return opts

# Preparation
opts = parse_opts()
lr_step='3,10,20,30,40,50,70,110,150,200,450,900,1500'

model_name = opts.model

lr = opts.lr
batch_size = opts.batch_size
momentum = opts.momentum
wd = opts.wd

lr_factor = opts.lr_factor
lr_steps = [int(s) for s in lr_step.split(',')] + [np.inf]

num_gpus = opts.num_gpus
num_workers = opts.num_workers
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = batch_size * max(num_gpus, 1)

jitter_param = 0.4
lighting_param = 0.1
resize_factor=1.5

transform_test = transforms.Compose([
    transforms.Resize(int(resize_factor * opts.input_sz)),
    # transforms.Resize(opts.input_sz, keep_ratio=True),
    transforms.CenterCrop(opts.input_sz),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    #metric = mx.metric._BinaryClassificationMetrics()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()

def setup_logger(log_file_path):
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    return logger

def save_params(net, best_acc, current_acc, epoch, prefix):
    current_acc = float(current_acc)
    if current_acc > best_acc:
        best_acc = current_acc
        net.save_parameters('{:s}_{:04d}_{:.4f}_best.params'.format(prefix, epoch, current_acc))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_acc))

def get_network(model, opts):
    network = get_model(model, pretrained=True)
    with network.name_scope():
        network.output = nn.Dense(opts.num_class)
        network.output.initialize(init.Xavier(), ctx=ctx)
        network.collect_params().reset_ctx(ctx)
        if opts.resume_params is not '':
            network.load_parameters(opts.resume_params, ctx=ctx)
    network.hybridize()
        #viz.plot_network(network, shape=(1, 3, 112, 112), save_prefix='test')
    return network

def test_network(model, params_path, val_path):
    finetune_net = get_network(model, opts)
    finetune_net.load_parameters(params_path)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    _, test_acc = test(finetune_net, val_data, ctx)
    print 'Test accuracy: '+str(test_acc)


def train(train_path, val_path, test_path):
    finetune_net = get_network(opts.model,opts)
    folder = opts.model+'_'+str(opts.input_sz)
    date_time = datetime.now().strftime('%Y-%m-%d_%H.%M')
    logger=setup_logger(os.path.join(folder,date_time,'train_log.log'))

    train_history = TrainingHistory(['training-error', 'validation-error'])

    transform_train = transforms.Compose([
        transforms.Resize(int(resize_factor * opts.input_sz)),
        transforms.RandomResizedCrop(opts.input_sz, scale=(0.4, 1)),
        # transforms.Resize(opts.input_sz, keep_ratio=True),
        # transforms.CenterCrop(opts.input_sz),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define DataLoader

    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)


    # Define Trainer
    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    metric_train = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    lr_counter = 0
    num_batch = len(train_data)

    print 'Begin finetuning', model_name, opts.input_sz
    # Start Training

    logger.info(opts)
    best_acc=0
    new_lr=trainer.learning_rate
    for epoch in range(epochs):
        if epoch == lr_steps[lr_counter]:
            new_lr=trainer.learning_rate*lr_factor
            trainer.set_learning_rate(new_lr)
            lr_counter += 1

        tic = time.time()
        metric_train.reset()
        btic = time.time()
        tic = time.time()
        train_loss=0
        for idx, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_bloss = 0
            train_bloss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            train_loss+=train_bloss
            metric_train.update(label, outputs)

            if opts.log_interval and not (idx + 1) % opts.log_interval:
                train_metric_name, train_metric_score = metric_train.get()
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\tloss: %f\t%s=%f\tlr=%f' % (
                    epoch, idx, batch_size * opts.log_interval / (time.time() - btic), train_bloss,
                    train_metric_name, train_metric_score, trainer.learning_rate))
                btic = time.time()

        _, train_acc = metric_train.get()
        train_loss /= num_batch

        _, val_acc = test(finetune_net, val_data, ctx)

        train_history.update([1 - train_acc, 1 - val_acc])
        train_history.plot(save_path=os.path.join(folder,date_time,'%s_history.png' % (model_name)))

        logger.info('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f | Speed: %.2f samples/sec | lr: %.8f' %
                 (epoch, train_acc, train_loss, val_acc, time.time() - tic, num_training_samples/(time.time()-tic), new_lr))
        val_acc = float(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            finetune_net.save_parameters(os.path.join(folder,date_time,'%s-%s-best.params' % (dataset, model_name)))
            trainer.save_states(os.path.join(folder, date_time, '%s-%s-best.states' % (dataset, model_name)))
            with open(os.path.join(folder,date_time,'best_map.log'), 'a') as f:
                f.write('{:04d}:\t{:.4f}\n'.format(epoch, val_acc))

    #_, test_acc = test(finetune_net, test_data, ctx)
    os.rename(folder,'{:s}_{:d}'.format(folder,int(10000*best_acc)))
    logger.info('[Finished]')

if __name__ == "__main__":
    #test_aug()
    train(train_path, test_path, test_path)
    #best_param_dir='arm_network_v3.3_112_8273/2019-04-21_07.21_8273'
    #test_network('arm_network_v3.3',os.path.join(best_param_dir,'best.params'),test_path)
