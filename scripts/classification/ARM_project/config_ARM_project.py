
#data_dir='/media/atsg/Data/datasets/SUN_ARM_project'
import os

pc='duycuongAI'
pc='300'


if(pc=='duycuongAI'):
    dataset_dir='/media/duycuong/Data/Dataset/ZaloAIChallenge2018/landmark'
    train_dir = os.path.join(dataset_dir, 'TrainVal/train')
    val_dir = os.path.join(dataset_dir, '/TrainVal/val')
    test_dir = os.path.join(dataset_dir, 'Test_Public')
else:
    dataset_dir='/media/atsg/Data/datasets/SUN_ARM_project'
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'test')
    test_dir = os.path.join(dataset_dir, 'landmark/Test_Public')

    # dataset_dir = '/media/atsg/Data/datasets/ImageNet/imagenet'
    # train_dir = os.path.join(dataset_dir, 'train')
    # val_dir = os.path.join(dataset_dir, 'val')
    # test_dir = os.path.join(dataset_dir, 'test')

classes = 2
model_name= 'arm_network_v3.5.2'  #'resnet18_v2'    #'resnext50_32x4d'
input_sz=112

#hyper parameters

dataset='ZaloAILandmark'
batch_size=32
epochs=200
log_interval=20
num_workers=6
base_lr=0.4
lr_decay=0.1
lr_mode='cosine'
lr_decay_epoch='40,60'
save_frequency=5

#training
resume_param=''
resume_state=''
resume_epoch=0

#testing
pretrained_param='ZaloAILandmark-resnext50_32x4d-20.params'
submission_prefix='38'

#data analyze
data_analyze_dir='data_analyze'
