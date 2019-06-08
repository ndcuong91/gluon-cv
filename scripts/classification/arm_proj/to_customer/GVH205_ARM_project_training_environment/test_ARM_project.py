import mxnet as mx
import numpy as np
import os, time, logging, argparse, shutil

from mxnet import gluon, image, init, nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from arm_network import get_arm_network
import config_ARM_project as config


model_version='v4.4'
folder=os.path.join('/home/atsg/PycharmProjects/gvh205/arm_project/model',model_version)
params_file = os.path.join(folder,'arm_v4.4_180_9101.params')

label=['clean_normal','messy_dirty']

model=config.model_name
ctx=[mx.gpu()]
num_workers=4
input_sz=config.input_sz
classes = config.classes
batch_size=config.batch_size

test_path = config.val_dir
resize_factor=config.resize_factor

def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        first_img= data[0][0].asnumpy()
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


def get_network(model, params):
    if('arm_network' in model):
        version=model.replace('arm_network_v','')
        network = get_arm_network(version, ctx)
        network.output = nn.Dense(classes)
        network.output.initialize(init.Xavier(), ctx=ctx)
        if params is not '':
            network.load_parameters(params, ctx=ctx, allow_missing=True, ignore_extra=True)
        network.hybridize()
    else:
        network = get_model(model, pretrained=True)
        with network.name_scope():
            network.output = nn.Dense( classes)
            network.output.initialize(init.Xavier(), ctx=ctx)
            network.collect_params().reset_ctx(ctx)
            network.hybridize()
    return network

def test_network(model, params_path, val_path):
    finetune_net = get_network(model, params_path)

    transform_test = transforms.Compose([
        transforms.Resize(int(resize_factor * input_sz)),
        # transforms.Resize(opts.input_sz, keep_ratio=True),
        transforms.CenterCrop(input_sz),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [1., 1., 1.])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    _, test_acc = test(finetune_net, test_data, ctx)
    print 'Test accuracy: '+str(test_acc)


if __name__ == "__main__":
    test_network(model,params_file,test_path)
