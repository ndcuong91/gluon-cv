import mxnet as mx
import numpy as np
import os, time, logging, argparse, shutil

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs, TrainingHistory, LRSequential, LRScheduler, viz
from gluoncv.model_zoo import get_model
import gluoncv as gcv
from datetime import datetime
import config_arm_project as config
import utils_classification as utils
from arm_network import get_arm_network

model=config.model_name
classes = config.classes
input_sz=config.input_sz
batch_size=config.batch_size
epochs=config.epochs
log_interval=config.log_interval
num_workers=config.num_workers
dataset=config.dataset
train_path = config.train_dir
test_path = config.val_dir
base_lr=config.base_lr
lr_decay=config.lr_decay
lr_decay_epoch=config.lr_decay_epoch
lr_mode=config.lr_mode
save_frequency=config.save_frequency

resume_param=config.resume_param
resume_state=config.resume_state
resume_epoch=config.resume_epoch
teacher=config.teacher
teacher_params=config.teacher_params


def parse_opts():
    parser = argparse.ArgumentParser(description='Transfer learning on dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str,default=model,
                        help='name of the pretrained model from model zoo.')
    parser.add_argument('-j', '--workers', dest='num_workers', default=num_workers, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='number of gpus to use, 0 indicates cpu only')
    parser.add_argument('--num_class', default=classes, type=int,
                        help='number of class')
    parser.add_argument('--num_epochs', type=int, default=epochs,
                        help='number of training epochs.')
    parser.add_argument('--input_sz', default=input_sz, type=int,
                        help='resolution of input size')
    parser.add_argument('-b', '--batch_size', default=batch_size, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning_rate', default=base_lr, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--log_interval', default=log_interval, type=int,
                        help='learning rate decay ratio')
    parser.add_argument('--save_frequency', type=int, default=save_frequency,
                        help='frequency of model saving.')

    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--lr_mode', type=str, default=lr_mode,
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr_decay', type=float, default=lr_decay,
                        help='decay rate of learning rate. default is 0.75.')
    parser.add_argument('--lr_decay_period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr_decay_epoch', type=str, default=lr_decay_epoch,
                        help='epochs at which learning rate decays. default is 40,60.')

    parser.add_argument('--resume_params', type=str, default=resume_param,
                        help='path of parameters to load from.')
    parser.add_argument('--resume_states', type=str, default=resume_state,
                        help='path of trainer state to load from.')
    parser.add_argument('--resume_epoch', type=int, default=resume_epoch,
                        help='epoch to resume training from.')

    #for distillation training
    parser.add_argument('--teacher', type=str, default=teacher, #None #'resnext50_32x4d'
                        help='teacher model for distillation training')
    parser.add_argument('--temperature', type=float, default=20,
                        help='temperature parameter for distillation teacher model')
    parser.add_argument('--hard-weight', type=float, default=0.5,
                        help='weight for the loss of one-hot label for distillation training')
    parser.add_argument('--label-smoothing', default=False,
                        help='use label smoothing or not in training. default is False.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--mode', type=str, default='hybrid',
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    opts = parser.parse_args()
    return opts

# Preparation
opts = parse_opts()

model_name = opts.model

lr = opts.lr
batch_size = opts.batch_size
momentum = opts.momentum
wd = opts.wd

num_gpus = opts.num_gpus
num_workers = opts.num_workers
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = batch_size * max(num_gpus, 1)

jitter_param = 0.4
lighting_param = 0.1
resize_factor=1.0

mean=[0.485, 0.456, 0.406]
#mean=[0, 0, 0]
std=[1, 1, 1]
#std=[0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(int(resize_factor * opts.input_sz)),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.Resize(int(resize_factor * opts.input_sz)),
    # transforms.Resize(opts.input_sz, keep_ratio=True),
    transforms.CenterCrop(opts.input_sz),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def test(net, val_data, ctx):
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, val_top1=acc_top1.get()
    _, val_top5=acc_top5.get()
    return val_top1, val_top5

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


def get_network(model, opts, frozen=False):
    if ('arm_network' in model):
        #ctx=[mx.cpu()]
        version = model.replace('arm_network_v', '')
        network = get_arm_network(version, ctx)
        if opts.resume_params is not '':
            network.load_parameters(opts.resume_params, ctx=ctx, allow_missing=True, ignore_extra=True)
        network.output = nn.Dense(opts.num_class)
        network.output.initialize(init.Xavier(), ctx=ctx)
        network.hybridize()
        # from gluoncv.utils import export_block
        # export_block('arm_network_v3.4.1', network, preprocess=True, layout='HWC')

        print('Done.')
        #viz.plot_network(network,shape=(1,3,112,112),save_prefix='test')
    else:
        network = get_model(model, pretrained=True)

        if (frozen):
            print 'Frozen'
            for param in network.collect_params().values():
                if (param.grad_req != 'null'):
                    print param.name
                    param.grad_req = 'null'
        else:
            print 'No frozen'

        with network.name_scope():
            network.output = nn.Dense(opts.num_class)
            network.output.initialize(init.Xavier(), ctx=ctx)

            # test_layer = network.output.collect_params()['resnetv20_dense1_weight']._data
            # print test_layer

            network.collect_params().reset_ctx(ctx)
            if opts.resume_params is not '':
                network.load_parameters(opts.resume_params, ctx=ctx)
        network.hybridize()
        #viz.plot_network(network, shape=(1, 3, 112, 112), save_prefix='test')
        return network
    return network


def test_network(model, params_path, val_path):
    finetune_net = get_network(model, opts)
    finetune_net.load_parameters(params_path)

    val_data = gluon.data.DataLoader(
        utils.ImageFolderDatasetCustomized(val_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    _, test_acc = test(finetune_net, val_data, ctx)
    print 'Test accuracy: '+str(test_acc)

def smooth(label, classes, eta=0.1):
    if isinstance(label, nd.NDArray):
        label = [label]
    smoothed = []
    for l in label:
        res = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
        smoothed.append(res)
    return smoothed

def train(train_path, test_path):
    finetune_net = get_network(opts.model,opts)
    if resume_param is '' and 'arm_network' in model:
        finetune_net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    folder = 'output/'+opts.model+'_'+str(opts.input_sz)+'_' + dataset
    date_time = datetime.now().strftime('%Y-%m-%d_%H.%M')
    logger=setup_logger(os.path.join(folder,date_time,'train_log.log'))

    train_history = TrainingHistory(['training-error', 'validation-error'])

    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)


    #lr_scheduler:
    lr_decay = opts.lr_decay
    lr_decay_period = opts.lr_decay_period
    if opts.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opts.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opts.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opts.warmup_epochs for e in lr_decay_epoch]
    num_training_samples=len(train_data._dataset)
    num_batches = num_training_samples // batch_size
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=opts.lr,
                    nepochs=opts.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(opts.lr_mode, base_lr=opts.lr, target_lr=0,
                    nepochs=opts.num_epochs - opts.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    if opts.teacher is not None and opts.hard_weight < 1.0:
        teacher_name = opts.teacher
        teacher = get_model(teacher_name, pretrained=False, classes=classes, ctx=ctx)
        teacher.load_parameters(teacher_params, ctx=ctx, allow_missing=True, ignore_extra=True)
        teacher.cast(opts.dtype)
        distillation = True
    else:
        distillation = False

    if opts.mode == 'hybrid':
        #finetune_net.hybridize(static_alloc=True, static_shape=True)
        if distillation:
            teacher.hybridize(static_alloc=True, static_shape=True)

    optimizer = 'sgd'
    optimizer_params = {'wd': opts.wd, 'momentum': opts.momentum, 'lr_scheduler': lr_scheduler}

    # Define Trainer
    trainer = gluon.Trainer(finetune_net.collect_params(), optimizer, optimizer_params)
    if opts.resume_states is not '':
        trainer.load_states(opts.resume_states)

    metric_train = mx.metric.Accuracy()

    if opts.label_smoothing:
        sparse_label_loss = False
    else:
        sparse_label_loss = True

    if distillation:
        L = gcv.loss.DistillationSoftmaxCrossEntropyLoss(temperature=opts.temperature,
                                                         hard_weight=opts.hard_weight,
                                                         sparse_label=sparse_label_loss)
    else:
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)
        #L = gluon.loss.SoftmaxCrossEntropyLoss()

    num_batch = len(train_data)

    print 'Begin training', model_name, opts.input_sz
    print 'Num samples in dataset:',num_training_samples
    # Start Training

    logger.info(opts)
    best_acc=0
    for epoch in range(opts.resume_epoch, opts.num_epochs):
        metric_train.reset()
        btic = time.time()
        tic = time.time()
        train_loss=0
        for idx, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            if opts.label_smoothing:
                hard_label = label
                label = smooth(label, classes)

            if distillation:
                teacher_prob = [nd.softmax(teacher(X.astype(opts.dtype, copy=False)) / opts.temperature) for X in data]
                #teacher_prob = [teacher(X.astype(opts.dtype, copy=False) / opts.temperature) for X in data]

            with ag.record():
                outputs = [finetune_net(X.astype(opts.dtype, copy=False)) for X in data]
                if distillation:
                    loss = [L(yhat.astype('float32', copy=False),
                              y.astype('float32', copy=False),
                              p.astype('float32', copy=False)) for yhat, y, p in zip(outputs, label, teacher_prob)]
                else:
                    loss = [L(yhat, y.astype(opts.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                    #outputs = [finetune_net(X) for X in data]
                    #loss = [L(yhat, y) for yhat, y in zip(outputs, label)]

            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_bloss = 0
            train_bloss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            train_loss+=train_bloss
            if opts.label_smoothing:
                metric_train.update(hard_label, outputs)
            else:
                metric_train.update(label, outputs)

            if opts.log_interval and not (idx + 1) % opts.log_interval:
                train_metric_name, train_metric_score = metric_train.get()
                logger.info('Epoch[%d] Batch [%d] \tSpeed: %f samples/sec\tloss: %f\t%s=%f\tlr=%f' % (
                    epoch, idx, batch_size * opts.log_interval / (time.time() - btic), train_bloss,
                    train_metric_name, train_metric_score, trainer.learning_rate))
                btic = time.time()
                # print test_layer

        _, train_acc = metric_train.get()
        train_loss /= num_batch

        val_acc_top1, val_acc_top5 = test(finetune_net, val_data, ctx)

        train_history.update([1 - train_acc, 1 - val_acc_top1])
        train_history.plot(save_path=os.path.join(folder,date_time,'%s_history.png' % (model_name)))

        logger.info('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc-top1: %.3f | time: %.1f | Speed: %.2f samples/sec | lr: %.8f' %
                 (epoch, train_acc, train_loss, val_acc_top1, time.time() - tic, num_training_samples/(time.time()-tic), trainer.learning_rate))
        val_acc_top1 = float(val_acc_top1)
        if val_acc_top1 > best_acc:
            best_acc = val_acc_top1
            finetune_net.save_parameters(os.path.join(folder,date_time,'%s-best-%d.params' % (model_name, epoch)))
            #trainer.save_states(os.path.join(folder, date_time, '%s-%s-best-%d.states' % (model_name, epoch)))
            with open(os.path.join(folder,date_time,'best_acc.log'), 'a') as f:
                f.write('{:04d}:\t{:.4f}\n'.format(epoch, val_acc_top1))

        if opts.save_frequency and epoch % opts.save_frequency == 0 and epoch>0:
            finetune_net.save_parameters(os.path.join(folder,date_time,'%s-%d.params' % (model_name, epoch)))
            #trainer.save_states(os.path.join(folder, date_time, '%s-%s-%d.states' % (model_name, epoch)))

    #_, test_acc = test(finetune_net, test_data, ctx)
    # os.rename(folder,'{:s}_{:d}'.format(folder,int(10000*best_acc)))
    logger.info('[Finished]')

if __name__ == "__main__":
    train(train_path, test_path)
