import mxnet as mx
import numpy as np
import os, time, logging, argparse, shutil

from mxnet import image, init, nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
import config_classification as config

input_sz=config.input_sz
num_class = config.classes
batch_size=config.batch_size
num_workers=config.num_workers
dataset=config.dataset
ctx=[mx.gpu()]

model_name=config.model_name
best_param_dir = os.path.join(model_name+'_'+str(input_sz),'2019-05-02_18.26')
model_snapshot_suffix='best'
model_snapshot=  dataset+'-'+model_name+'-'+ model_snapshot_suffix

resize_factor=1.5
jitter_param = 0.4
lighting_param = 0.1


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

re_map=[0,1,10,100,101,102,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

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

def classify_img(net, file_path, topk=3, test_time_augment=1):
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
    topk_pred_idx = pred.argsort()[-1:topk_axis:-1]
    return pred, topk_pred_idx

def submission(model_name, params_path, print_process=100, test_time_augment=1, topk=3):
    finetune_net = get_network(model_name)
    finetune_net.load_parameters(params_path)

    count=0
    result='id,predicted\n'

    files_list = get_list_file_in_folder(test_dir)
    files_list=sorted(files_list)
    for f in files_list:
        result += f.replace('.jpg', '') + ','
        if (count % print_process == 0):
            print('Tested: ' + str(count) + " files")
        #print f
        file_path = os.path.join(os.path.join(test_dir, f))
        pred, topk_pred = classify_img(finetune_net,file_path,topk,test_time_augment)
        for k in range(len(topk_pred)):
            if (k < 2):
                result += str(re_map[topk_pred[k]]) + ' '
            else:
                result += str(re_map[topk_pred[k]]) + '\n'

        count+=1

    with open('13_submission_CuongND_resnet152_v2_224_val_acc_9396.csv', 'w') as file:
        file.write(result)

def classify_dir(model_name,params_path, print_process=50, test_time_augment=1, topk=3):
    finetune_net = get_network(model_name)
    finetune_net.load_parameters(params_path)

    topk_pred_result=[]
    for n in range(num_class):
        topk_pred_result.append(0)

    count=0
    files_list = get_list_file_in_folder(test_dir)
    files_list=sorted(files_list)
    for f in files_list:
        if (count % print_process == 0):
            print('Tested: ' + str(count) + " files")
        #print f
        file_path = os.path.join(os.path.join(test_dir, f))

        pred, topk_pred=classify_img(finetune_net,file_path,topk,test_time_augment)
        for k in range(len(topk_pred)):
            remap_id = re_map[topk_pred[k]]
            topk_pred_result[remap_id] += pred[topk_pred[k]]

            # if(max_prob>0.65):
            #     #shutil.copy(file_path, os.path.join(result_test_folder,str(ind.asscalar()),f))
            #     high_score+=1

        count+=1

    topk_result = ''
    for i in range(num_class):
        topk_result += str(int(topk_pred_result[i])) + '\n'
    with open('result_public_test_top' + str(topk) + '_prob.txt', 'w') as file:
        file.write(topk_result)
    #print 'There are',high_score,'file has max_prob >0.65 over',count,'files'

def test(net, data_dir, write_output=False,output_file_name='result', topk=5, test_time_augment=1):
    count=0
    true_pred=0
    true_class=0
    topk_axis=-1-topk
    print 'Predict with TTA =',test_time_augment,',topk =',topk

    result=''
    topk_pred_result=[]
    for n in range(num_class):
        topk_pred_result.append(0)

    for cls_int in range(num_class):
        cls=str(cls_int)
        count_cls = 0
        true_pred_cls = 0
        files_list = get_list_file_in_folder(os.path.join(data_dir,cls))
        for f in files_list:
            #print file
            file_path=os.path.join(os.path.join(data_dir,cls,f))
            pred, topk_pred = classify_img(net, file_path, topk, test_time_augment)
            for k in range(len(topk_pred)):
                remap_id=re_map[topk_pred[k]]
                topk_pred_result[remap_id]+=pred[topk_pred[k]]
                if (remap_id==cls_int):
                    true_pred_cls+=1
                    true_pred+=1
            count_cls+=1
            count+=1

        result += str(true_pred_cls) + '\n'
        print 'Class: ',cls,',true_pred',true_pred_cls,',total',count_cls,',top'+str(topk), float(true_pred_cls) / float(count_cls)
        true_class+=1
    print 'True pred:',true_pred,', Total file:',count,'top'+str(topk)+' accuracy: ',float(true_pred)/float(count)


    if(write_output):
        with open(output_file_name+'_true_pred_top'+str(topk)+'.txt', 'w') as file:
            file.write(result)
        topk_result=''
        for i in range(num_class):
            topk_result+=str(int(topk_pred_result[i]))+'\n'
        with open(output_file_name+'_top'+str(topk)+'_prob.txt', 'w') as file:
            file.write(topk_result)

    # if (topk == 1):
    #     metric = mx.metric.Accuracy()
    # else:
    #     metric = mx.metric.TopKAccuracy(top_k=topk)
    #
    # for i, batch in enumerate(val_data):
    #     if (i % 50 == 0 and i > 0):
    #         print 'Tested:', i, 'batches'
    #     data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    #     label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    #     outputs = [net(X) for X in data]
    #     metric.update(label, outputs)
    #
    # _,test_acc=metric.get()
    #
    # print 'Accuracy (top '+str(topk)+'):',str(test_acc)
    # print 'Error (top '+str(topk)+'):',str(1-test_acc)

    return

def get_network(model_name):
    network = get_model(model_name, pretrained=True)

    with network.name_scope():
        network.output = nn.Dense(num_class)
        network.output.initialize(init.Xavier(), ctx=ctx)
        network.collect_params().reset_ctx(ctx)
        network.hybridize()
    return network

def test_network(model_name, params_path, data_dir, test_time_augment=1):
    print 'Start test network:',model_name,'with params:',params_path
    finetune_net = get_network(model_name)
    finetune_net.load_parameters(params_path)

    #test(finetune_net, data_dir, write_output=True,output_file_name='result_val', topk=1, test_time_augment=test_time_augment)
    test(finetune_net, data_dir, write_output=True,output_file_name='result_public', topk=5, test_time_augment=test_time_augment)
    #test(finetune_net, data_dir, write_output=True,output_file_name='result_val', topk=5, test_time_augment=test_time_augment)
    print 'Finish'

if __name__ == "__main__":
    #test_network(model_name,os.path.join(best_param_dir,model_snapshot+'.params'),val_dir, 1)
    #submission(model_name,os.path.join(best_param_dir,model_snapshot+'.params'),test_time_augment=1)
    classify_dir(model_name,os.path.join(best_param_dir,model_snapshot+'.params'),test_time_augment=1)