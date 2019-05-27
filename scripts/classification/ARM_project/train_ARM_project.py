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
from arm_network import get_arm_network
import cv2
import config_ARM_project as config

data_dir=config.data_dir
#data_dir='/media/atsg/Data/CuongND/challenge/zaloAIchallenge/landmark/data/TrainVal'
model='resnet18_v2'
#model='arm_network_v3.5.2'
classes = config.classes
input_sz=config.input_sz
batch_size=config.batch_size

#because of small dataset, we use test set as validation set
train_path = os.path.join(data_dir, 'train')
test_path = os.path.join(data_dir, 'test')

def parse_opts():
    parser = argparse.ArgumentParser(description='Transfer learning on SUN_ARM_project dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str,default=model,
                        help='name of the pretrained model from model zoo.')
    parser.add_argument('-j', '--workers', dest='num_workers', default=8, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-gpus', default=1, type=int,
                        help='number of gpus to use, 0 indicates cpu only')
    parser.add_argument('--num_class', default=classes, type=int,
                        help='number of class')
    parser.add_argument('--input_sz', default=input_sz, type=int,
                        help='resolution of input size')
    parser.add_argument('-b', '--batch-size', default=batch_size, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-factor', default=0.75, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--log_interval', default=50, type=int,
                        help='learning rate decay ratio')
    opts = parser.parse_args()
    return opts

# Preparation
opts = parse_opts()

if 'arm_network' in opts.model:
    epochs=2000
    lr_step='10,20,30,40,50,70,110,150,200,500,1000'
else:
    epochs=5000
    lr_step='10,20,30,40,50,70,110,150,200,450,900,1500'
if 'imagenet' in data_dir:
    epochs=300
    lr_step='40,60'

model_name = opts.model
print model_name
print opts.input_sz

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
    if('arm_network' in model):
        version=model.replace('arm_network_v','')
        network=get_arm_network(version,opts.num_class,ctx)
        network.hybridize()
        #viz.plot_network(network,shape=(1,3,112,112),save_prefix='test')
    else:
        network = get_model(model, pretrained=True)
        with network.name_scope():
            network.output = nn.Dense(opts.num_class)
            network.output.initialize(init.Xavier(), ctx=ctx)
            network.collect_params().reset_ctx(ctx)
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

def test_RecordIO():
    from mxnet.io import ImageRecordIter
    train_data = ImageRecordIter(
        path_imgrec=os.path.join('/media/atsg/Data/datasets/SUN_ARM_project', 'train.rec'),
        path_imgidx=os.path.join('/media/atsg/Data/datasets/SUN_ARM_project', 'train.idx'),

        data_shape=(3, 112, 112),
        batch_size=48,
        shuffle=True
    )

    for batch in train_data:
        data=batch.data[0]
        label=batch.label[0]
        kk=1


def save_aug(transformed_data,save_prefix='test'):
    sample_data = transformed_data.asnumpy().transpose(1, 2, 0)
    scale = np.array([58.395, 57.12, 57.375])
    mean = np.array([123.675, 116.28, 103.53])
    image = cv2.cvtColor((np.round(sample_data*scale+mean)).astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join('aug_samples',save_prefix+'.jpg'),image)



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

    test_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)


    # Define Trainer
    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    lr_counter = 0
    num_batch = len(train_data)
    num_training_samples=len(train_data._dataset)

    print 'Begin training'
    print 'Num samples in dataset:',num_training_samples
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
        train_loss = 0
        metric.reset()
        btic = time.time()
        for idx, batch in enumerate(train_data):

            #print i,
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            #save_aug(data[0][0],str(i)+'_'+str(epoch))
            with ag.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]

            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            metric.update(label, outputs)

            if opts.log_interval and not (idx + 1) % opts.log_interval:
                train_metric_name, train_metric_score = metric.get()
                logger.info('Epoch[%d] Batch [%d] \tSpeed: %f samples/sec\t%s=%f\tlr=%f' % (
                    epoch, idx, batch_size * opts.log_interval / (time.time() - btic),
                    train_metric_name, train_metric_score, trainer.learning_rate))
                btic = time.time()

        #print
        _, train_acc = metric.get()
        train_loss /= num_batch

        _, test_acc = test(finetune_net, test_data, ctx)

        train_history.update([1 - train_acc, 1 - test_acc])
        train_history.plot(save_path=os.path.join(folder,date_time,'%s_history.png' % (model_name)))

        logger.info('[Epoch %d] Train-acc: %.3f, loss: %.3f | Test-acc: %.3f | time: %.1f | Speed: %.2f samples/sec | lr: %.8f' %
                 (epoch, train_acc, train_loss, test_acc, time.time() - tic, num_training_samples/(time.time()-btic), new_lr))
        test_acc = float(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            finetune_net.save_parameters(os.path.join(folder,date_time,'best.params'))
            with open(os.path.join(folder,date_time,'best_map.log'), 'a') as f:
                f.write('{:04d}:\t{:.4f}\n'.format(epoch, test_acc))

    #_, test_acc = test(finetune_net, test_data, ctx)
    os.rename(folder,'{:s}_{:d}'.format(folder,int(10000*best_acc)))
    logger.info('[Finished]')

if __name__ == "__main__":
    #test_RecordIO()
    #test_aug()
    train(train_path, test_path, test_path)
    #best_param_dir='arm_network_v3.3_112_8273/2019-04-21_07.21_8273'
    #test_network('arm_network_v3.3',os.path.join(best_param_dir,'best.params'),test_path)
