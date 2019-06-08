import os

dataset_dir='dataset'
dataset='imagenet' #dataset3_resize300
if(dataset=='new_dataset1_resize300' or dataset=='dataset3_resize300'):
    train_dir = os.path.join(dataset_dir, dataset, 'train')
    val_dir = os.path.join(dataset_dir, dataset, 'test') #val set is also test set
    classes = 2

if(dataset=='imagenet'):
    train_dir = '/home/atsg/.mxnet/datasets/imagenet/train'
    val_dir = '/home/atsg/.mxnet/datasets/imagenet/val'
    classes = 1000

model_name= 'arm_network_v4.4'

#hyper parameters
resize_factor=1.5
batch_size=128
log_interval=200
epochs=2000
num_workers=8
base_lr=0.01
lr_decay=0.7
lr_mode='step'
lr_decay_epoch='10,20,30,40,50,70,110,150,200,500,900,1400'
if (model_name=='arm_network_v3.4'):
    input_sz=112
    resume_param = 'pretrained/ImageNet_v3.4_3452.params'
if (model_name=='arm_network_v3.5.2'):
    input_sz=112
    resume_param = 'pretrained/ImageNet_v3.5.2_3440.params'
if (model_name=='arm_network_v4.1'):
    input_sz=160
    resume_param = 'pretrained/ImageNet_v4.1_4352.params'
if (model_name=='arm_network_v4.2'):
    input_sz=180
    resume_param = 'pretrained/ImageNet_v4.2_4551.params'
if (model_name=='arm_network_v4.3'):
    input_sz=180
    resume_param = 'pretrained/ImageNet_v4.3_4150.params'
# if (model_name=='arm_network_v4.4'):
#     input_sz=160
#     resume_param = 'pretrained/ImageNet_v4.4_3952_160.params'
if (model_name=='arm_network_v4.4'):
    input_sz=180
    resume_param = 'pretrained/ImageNet_v4.4_3952_160.params'
if (model_name=='arm_network_v4.4.1'):
    input_sz=180
    resume_param = 'pretrained/ImageNet_v4.4_3952_160.params'

resume_state = ''
resume_epoch = 0
save_frequency=10
if(dataset=='imagenet'):
    #resume_param=''
    base_lr=0.01
    lr_decay=0.3
    lr_decay_epoch='10,18,28,40,60,90,140,200,500,900,1400'
    save_frequency=5
    resize_factor=1.0



