
#data_dir='/media/atsg/Data/datasets/SUN_ARM_project'
import os

pc='duycuongAI'
pc='300'


if(pc=='duycuongAI'):
    dataset_dir='/media/duycuong/Data/Dataset/ZaloAIChallenge2018/landmark'
    train_dir = os.path.join(dataset_dir, 'TrainVal/train')
    val_dir = os.path.join(dataset_dir, '/TrainVal/val')
    test_dir = os.path.join(dataset_dir, 'Test_Public')

if(pc=='300'):
    dataset_dir='/media/atsg/Data/datasets/SUN_ARM_project'
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'test')
    test_dir = os.path.join(dataset_dir, 'landmark/Test_Public')

if(pc=='370'):
    dataset_dir='/media/atsg/Data/datasets/SUN_ARM_project'
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'test')
    test_dir = os.path.join(dataset_dir, 'landmark/Test_Public')

    # dataset_dir = '/media/atsg/Data/datasets/ImageNet/imagenet'
    # train_dir = os.path.join(dataset_dir, 'train')
    # val_dir = os.path.join(dataset_dir, 'val')
    # test_dir = os.path.join(dataset_dir, 'test')

classes = 2
model_name= 'arm_network_v4.5.2'  #'resnet18_v2'    #'resnext50_32x4d'
input_sz=160

#hyper parameters
dataset='ImageNet'
batch_size=64
epochs=1500
log_interval=200
num_workers=6
training=True

if(training==True):
    base_lr=0.01
    lr_decay=0.75
    lr_mode='step'
    lr_decay_epoch='10,20,30,40,50,70,110,150,200,500,1000'
    resume_param = ''
    resume_state = ''
    resume_epoch = 0
else: #finetuning
    base_lr=0.0001
    lr_decay=0.5
    lr_mode='step'
    lr_decay_epoch='10,20,30,40,50,70,110,150,200,500,1000'
    resume_param = ''
    resume_state = ''
    resume_epoch = 0

save_frequency=5

#testing
pretrained_param='ZaloAILandmark-resnext50_32x4d-20.params'
submission_prefix='38'

#data analyze
data_analyze_dir='data_analyze'