import os

pc='duycuongAI'
pc='300'

dataset='new_dataset1_resize300' #dataset3_resize300
if(pc=='duycuongAI'):
    dataset_dir='/media/duycuong/Data/Dataset/ZaloAIChallenge2018/landmark'
    train_dir = os.path.join(dataset_dir, 'TrainVal/train')
    val_dir = os.path.join(dataset_dir, '/TrainVal/val')
    test_dir = os.path.join(dataset_dir, 'Test_Public')

if(pc=='300'):
    dataset_dir='/home/atsg/PycharmProjects/gvh205/arm_proj/to_customer/GVH205_ARM_project_training_environment/dataset/'+dataset
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'test')

classes = 2
model_name= 'arm_network_v4.3'# 'resnext50_32x4d'

#hyper parameters
batch_size=32
epochs=2500
log_interval=200
num_workers=8

training=True
resume_param = ''
resume_state = ''
resume_epoch = 0

if ('arm_network_v3.4' in model_name):
    input_sz=112
    resume_param = 'pretrained/ImageNet_v3.4_3452.params'
    #resume_param = 'arm_network_v3.4.1-best-52.params'
elif ('arm_network_v3.5.2' in model_name):
    input_sz=112
    resume_param = 'pretrained/ImageNet_v3.5.2_3440.params'
elif ('arm_network_v4.1' in model_name):
    input_sz=160
    resume_param = 'pretrained/ImageNet_v4.1_4352.params'
elif ('arm_network_v4.2' in model_name):
    input_sz=180
    resume_param = 'pretrained/ImageNet_v4.2_4551.params'
elif ('arm_network_v4.3' in model_name):
    input_sz=180
    resume_param = 'pretrained/ImageNet_v4.3_4150.params'
elif ('arm_network_v4.4' in model_name):
    input_sz=180
    resume_param = 'pretrained/ImageNet_v4.4_4000_180.params'
else:
    input_sz=160


if(training==True):
    base_lr=0.01
    lr_decay=0.7
    lr_mode='step'
    lr_decay_epoch='10,20,30,40,50,70,110,150,200,500,900,1400'
else: #finetuning
    base_lr=0.0001
    lr_decay=0.5
    lr_mode='step'
    lr_decay_epoch='500,1000'
    resume_param = 'resnext50_32x4d_180/2019-06-18_16.30.params'
    resume_state = ''
    resume_epoch = 0

#resume_param = 'resnext50_32x4d_180/2019-06-18_16.30/new_dataset1_resize300-resnext50_32x4d-best-48.params'
#resume_state = 'resnext50_32x4d_180/2019-06-18_16.30/new_dataset1_resize300-resnext50_32x4d-best-48.states'
teacher=''
save_frequency=200
