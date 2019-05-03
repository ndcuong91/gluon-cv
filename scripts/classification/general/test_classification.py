import mxnet as mx
import numpy as np
import os, time, logging, argparse, shutil

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.model_zoo import get_model
from datetime import datetime

data_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal/'
model='resnext50_32x4d'
best_param_dir = 'resnext50_32x4d_224/2019-05-02_18.26'
classes = 103
input_sz=224
batch_size=32
ctx=[mx.gpu()]
num_workers=4
model_name='ZaloAILandmark-resnext50_32x4d-best'

val_dir = os.path.join(data_dir, 'val')
resize_factor=1.5

transform_test = transforms.Compose([
    transforms.Resize(int(resize_factor * input_sz)),
    # transforms.Resize(opts.input_sz, keep_ratio=True),
    transforms.CenterCrop(input_sz),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

re_map=[0,1,10,100,101,102,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir


def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def submission(net,params_path, print_process=100):
    finetune_net = get_network(model)
    finetune_net.load_parameters(params_path)

    test_folder='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public'
    count=0
    result='id,predicted\n'

    files_list = get_list_file_in_folder(test_folder)
    files_list=sorted(files_list)
    for f in files_list:
        result += f.replace('.jpg', '') + ','
        if (count % print_process == 0):
            print('Tested: ' + str(count) + " files")
        #print f
        file_path = os.path.join(os.path.join(test_folder, f))
        try:
            img = image.imread(file_path)
            img = transform_test(img)
            img_gpu = img.copyto(mx.gpu())
            outputs = finetune_net(img_gpu.expand_dims(axis=0))
            topk = outputs.asnumpy().flatten().argsort()[-1:-4:-1]
            for k in range(len(topk)):
                if (k < 2):
                    result += str(re_map[topk[k]]) + ' '
                else:
                    result += str(re_map[topk[k]]) + '\n'
        except:
            print 'damaged file:',f
            result += '1 87 25\n'
            pass

        count+=1

    with open('submission_CuongND_resnext50_32x4d_3_mxnet.csv', 'w') as file:
        file.write(result)


def test(net, val_data, ctx, topk=5): #0.989236509759 top5  #0.982204362801 top3
    if(topk==1):
        metric = mx.metric.Accuracy()
    else:
        metric = mx.metric.TopKAccuracy(top_k=topk)

    for i, batch in enumerate(val_data):
        if(i%50==0 and i>0):
            print 'Tested:',i,'batches'
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    # class_folder=get_list_dir_in_folder(test_path)
    # class_folder=sorted(class_folder)
    # count=0
    # true_pred=0
    # true_class=0
    # for cls in class_folder:
    #     print cls
    #     files_list = get_list_file_in_folder(os.path.join(test_path,cls))
    #     for f in files_list:
    #         #print file
    #         file_path=os.path.join(os.path.join(test_path,cls,f))
    #         img = image.imread(file_path)
    #         img = transform_test(img)
    #         img_gpu = img.copyto(mx.gpu())
    #         outputs = net(img_gpu.expand_dims(axis=0))
    #         topk= outputs.asnumpy().flatten().argsort()[-1:-4:-1]
    #         for k in range(len(topk)):
    #             remap_id=re_map[topk[k]]
    #             if (remap_id==int(cls)):
    #                 true_pred+=1
    #
    #         count+=1
    #     true_class+=1
    #     print 'top3 accuracy for class: ',true_class,' is: ', float(true_pred) / float(count)
    # print 'True pred:',true_pred,', Total file:',count,'top3 accuracy: ',float(true_pred)/float(count)
    _,test_acc=metric.get()

    print 'Accuracy (top '+str(topk)+'):',str(test_acc)
    print 'Error (top '+str(topk)+'):',str(1-test_acc)
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


def get_network(model):
    network = get_model(model, pretrained=True)
    with network.name_scope():
        network.output = nn.Dense(classes)
        network.output.initialize(init.Xavier(), ctx=ctx)
        network.collect_params().reset_ctx(ctx)
        network.hybridize()
    return network

def test_network(model, params_path, val_path):
    print 'Start test network:',model,'with params:',params_path
    finetune_net = get_network(model)
    finetune_net.load_parameters(params_path)

    # transform_test = transforms.Compose([
    #     transforms.Resize(input_sz),
    #     # transforms.Resize(opts.input_sz, keep_ratio=True),
    #     transforms.CenterCrop(input_sz),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    test_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    test(finetune_net, test_data, ctx, topk=1)
    test(finetune_net, test_data, ctx, topk=3)
    test(finetune_net, test_data, ctx, topk=5)
    print 'Finish'


if __name__ == "__main__":
    test_network(model,os.path.join(best_param_dir,model_name+'.params'),val_dir)
    #submission(model,os.path.join(best_param_dir,model_name+'.params'))
