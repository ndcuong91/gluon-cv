import os

pc='duycuongAI'
pc='300'


if(pc=='duycuongAI'):
    dataset_dir='/media/duycuong/Data/Dataset/ZaloAIChallenge2018/landmark'
    train_dir = os.path.join(dataset_dir, 'TrainVal/train')
    val_dir = os.path.join(dataset_dir, '/TrainVal/val')
    test_dir = os.path.join(dataset_dir, 'Test_Public')
else:
    dataset_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark'
    train_dir = os.path.join(dataset_dir, 'TrainVal1_fixed_class2/train')
    val_dir = os.path.join(dataset_dir, 'TrainVal1_fixed_class2/val')
    test_dir = os.path.join(dataset_dir, 'landmark/Test_Public')

    # dataset_dir = '/media/atsg/Data/datasets/ImageNet/imagenet'
    # train_dir = os.path.join(dataset_dir, 'train')
    # val_dir = os.path.join(dataset_dir, 'val')
    # test_dir = os.path.join(dataset_dir, 'test')

classes = 103
model_name= 'resnext50_32x4d'  #'resnet18_v2'    #'resnext50_32x4d'
input_sz=224

#hyper parameters

dataset='ZaloAILandmark'
batch_size=32
epochs=200
log_interval=200
num_workers=6
base_lr=0.01
lr_decay=0.75
lr_mode='step'
lr_decay_epoch='10,20,30,50,80,110,150,200,450,900,1500'
save_frequency=5

#training
resume_param='densenet161_224/2019-05-28_21.24/ZaloAILandmark-densenet161-best-21.params'
resume_state='densenet161_224/2019-05-28_21.24/ZaloAILandmark-densenet161-best-21.states'
resume_epoch=22

#testing
pretrained_param='resnext50_32x4d_224/2019-05-24_16.33/ZaloAILandmark-resnext50_32x4d-best-59.params'
submission_prefix='38'

#data analyze
data_analyze_dir='data_analyze'
