import mxnet as mx
import numpy as np
import os, time, logging, argparse, shutil
from mxnet import image, init, nd, gluon, ndarray
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
import config_classification as config
import utils_classification as utils

input_sz=config.input_sz
num_class = config.classes
batch_size=config.batch_size
num_workers=config.num_workers
dataset=config.dataset
ctx=[mx.gpu()]

model_name=config.model_name
pretrained_param = config.pretrained_param
test_data_dir = config.test_dir
submission_file=config.submission_file

resize_factor=1.5
jitter_param = 0.4
lighting_param = 0.1

seeds=[233,33,39,52,65]

val_dir = config.val_dir
test_dir = config.test_dir

transform_test = transforms.Compose([
    transforms.Resize(int(resize_factor * input_sz)),
    # transforms.Resize(opts.input_sz, keep_ratio=True),
    transforms.CenterCrop(input_sz),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


transform_test_TTA = transforms.Compose([
    transforms.Resize(int(resize_factor * input_sz)),
    # transforms.Resize(opts.input_sz, keep_ratio=True),
    transforms.CenterCrop(input_sz),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

map_from_testing_to_training=[0,1,10,100,101,102,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,58,59,6,60,61,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,76,77,78,79,8,80,81,82,83,84,85,86,87,88,89,9,90,91,92,93,94,95,96,97,98,99]
map_from_training_to_testing=[0,1,15,26,37,48,59,70,81,92,2,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58,60,61,62,63,64,65,66,67,68,69,71,72,73,74,75,76,77,78,79,80,82,83,84,85,86,87,88,89,90,91,93,94,95,96,97,98,99,100,101,102,3,4,5]

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

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


def get_network_with_pretrained(model_name, params_path):
    network = get_model(model_name, pretrained=True)

    with network.name_scope():
        network.output = nn.Dense(num_class)
        network.output.initialize(init.Xavier(), ctx=ctx)
        network.collect_params().reset_ctx(ctx)
        network.hybridize()
    network.load_parameters(params_path)
    return network

def classify_img(net, file_path, topk=3, test_time_augment=2, print_data=False):
    if(print_data):
        print file_path
    img = image.imread(file_path)
    topk_axis=-1-topk
    if (test_time_augment == 1):
        img_transform = transform_test(img)
        img_gpu = img_transform.copyto(mx.gpu())
        outputs = net(img_gpu.expand_dims(axis=0))
        pred = outputs.asnumpy().flatten()
    else:
        preds = []
        for n in range(test_time_augment):
            img_transform = transform_test_TTA(img)
            img_gpu = img_transform.copyto(mx.gpu())
            outputs = net(img_gpu.expand_dims(axis=0))
            preds.append(outputs.asnumpy().flatten())
        pred = np.mean(preds, axis=0)

    pred = nd.softmax(nd.array(pred)).asnumpy()
    remap_pred= pred[map_from_training_to_testing]

    remap_topk_pred_idx = remap_pred.argsort()[-1:topk_axis:-1]

    if(print_data):
        for i in range(topk):
            print 'Class:',remap_topk_pred_idx[i],', Prob:',remap_pred[remap_topk_pred_idx[i]]

    return remap_pred, remap_topk_pred_idx

def submission(net, test_time_augment=1, topk=3):
    print 'Make submission with network:',model_name,'with params:',pretrained_param
    print 'TTA =',test_time_augment,', topk =',topk,', batch_size =',batch_size

    name, topk_labels, topk_probs= classify_dir(net,test_dir,[mx.gpu()],topk=topk, test_time_augment=test_time_augment, sub_class=False)

    samples=name.shape[0]
    print 'data_dir: ',test_dir,', num samples =',samples
    result='id,predicted\n'
    for i in range(samples):
        result +=str(name[i]) + ','
        for k in range(topk):
            if (k < 2):
                result += str(topk_labels[i][k]) + ' '
            else:
                result += str(topk_labels[i][k]) + '\n'

    with open(submission_file, 'w') as file:
        file.write(result)
        print 'Save submission file to:',submission_file

def classify_dir_with_subclass(net, ctx, data_dir, seed=233):
    print 'Classify_dir_with_subclass. seed:',seed
    print 'data_dir:',data_dir
    mx.random.seed(seed)
    val_data = gluon.data.DataLoader(
        utils.ImageFolderDatasetCustomized(data_dir).transform_first(transform_test_TTA),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    for i, batch in enumerate(val_data):
        if (i % 50 == 0 and i > 0):
            print i, 'batches'
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        name = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)

        outputs=[]
        for y in data:
            outputs.append(net(y))

        if(i==0):
            total_label=label[0]
            total_name=name[0]
            total_output=(nd.softmax(outputs[0],axis=1)).asnumpy().astype('float32')
        else:
            total_label = ndarray.concat(total_label, label[0], dim=0)
            total_name = ndarray.concat(total_name, name[0], dim=0)
            total_output=np.concatenate((total_output, (nd.softmax(outputs[0],axis=1)).asnumpy().astype('float32')), axis=0)

    print
    return total_name.asnumpy(),total_label.asnumpy(),total_output

def classify_dir_wo_subclass(net, ctx, data_dir, seed=233):
    print 'classify_dir_wo_subclass. seed:',seed
    print 'data_dir:',data_dir
    mx.random.seed(seed)
    test_data = gluon.data.DataLoader(
        utils.ImageFolderDatasetCustomized(data_dir,sub_class_inside=False).transform_first(transform_test_TTA),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    for i, batch in enumerate(test_data):
        if (i % 50 == 0 and i > 0):
            print 'Tested:', i, 'batches'
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        name = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        outputs=[]
        for y in data:
            outputs.append(net(y))

        if(i==0):
            total_name=name[0]
            total_output=(nd.softmax(outputs[0],axis=1)).asnumpy().astype('float32')
        else:
            total_name = ndarray.concat(total_name, name[0], dim=0)
            total_output=np.concatenate((total_output, (nd.softmax(outputs[0],axis=1)).asnumpy().astype('float32')), axis=0)

    print
    return total_name.asnumpy(),total_output

def classify_dir(net,data_dir, ctx=[mx.gpu()], topk=3, test_time_augment=2, sub_class=True):
    print 'Test network:',model_name,'with params:',pretrained_param
    print 'TTA =',test_time_augment,',topk =',topk

    list_pred=[]
    for n in range(test_time_augment):
        if(sub_class):
            name, label, predict= classify_dir_with_subclass(net,ctx,data_dir, seeds[n])
        else:
            name, predict= classify_dir_wo_subclass(net,ctx,data_dir, seeds[n])
        list_pred.append(predict.flatten())
    samples=name.shape[0]
    pred = np.mean(list_pred, axis=0)  #tta average

    pred=pred.reshape(samples,-1)
    numpy_arr = (nd.softmax(nd.array(pred), axis=1)).asnumpy().astype('float32')

    total_labels = np.argsort(-numpy_arr, axis=1)
    topk_probs = -np.sort(-numpy_arr, axis=1)
    topk_labels= total_labels[:, 0:topk]
    topk_probs= topk_probs[:, 0:topk]

    if (sub_class):
        return name, label,topk_labels,topk_probs
    else:
        return name, topk_labels,topk_probs

def get_result(name, label, topk_labels, ):
    samples=len(name)

    class_true_pred=[]
    class_total_sample=[]
    topk=len(topk_labels[0])

    for n in range(num_class):
        class_true_pred.append(0)
        class_total_sample.append(0)


    for i in range(samples):
        cls=label[i]
        for idx in range(topk):
            if(map_from_testing_to_training[topk_labels[i][idx]]==cls):
                class_true_pred[cls]+=1
        class_total_sample[cls]+=1

    total_true_pred=0
    total_sample=0
    for n in range(num_class):
        total_true_pred+=class_true_pred[n]
        total_sample+=class_total_sample[n]
        print 'Class: ',n,',true_pred',class_true_pred[n],',total',class_total_sample[n],',top'+str(topk), float(class_true_pred[n]) / float(class_total_sample[n])

    print 'True pred:',total_true_pred,', Total file:',total_sample,'top'+str(topk)+' accuracy: ',float(total_true_pred)/float(total_sample)

    # if(write_output):
    #     with open(output_file_name+'_true_pred_top'+str(topk)+'.txt', 'w') as file:
    #         file.write(result)
    #     topk_result=''
    #     for i in range(num_class):
    #         topk_result+=str(int(topk_pred_result[i]))+'\n'
    #     with open(output_file_name+'_top'+str(topk)+'_prob.txt', 'w') as file:
    #         file.write(topk_result)

if __name__ == "__main__":
    finetune_net = get_network_with_pretrained(model_name, pretrained_param)
    begin_time=time.time()
    name, label, topk_labels, topk_probs= classify_dir(finetune_net,val_dir,[mx.gpu()],test_time_augment=2)
    get_result(name, label, topk_labels)
    #submission(finetune_net,test_time_augment=5)
    #classify_img(finetune_net,'/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/TrainVal/train/2/16828.jpg',print_data=True)
    print 'Total time=',time.time() - begin_time
    print 'Finish'