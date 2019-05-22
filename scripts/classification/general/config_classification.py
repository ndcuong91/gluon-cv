import os

pc='duycuongAI'
#pc='300'


if(pc=='duycuongAI'):
    dataset_dir='/media/duycuong/Data/Dataset/ZaloAIChallenge2018/landmark'
    train_dir = os.path.join(dataset_dir, 'TrainVal/train')
    val_dir = os.path.join(dataset_dir, '/TrainVal/val')
    test_dir = os.path.join(dataset_dir, 'Public')
else:
    dataset_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018'
    train_dir = os.path.join(dataset_dir, 'landmark/TrainVal1_fixed_class2/train')
    val_dir = os.path.join(dataset_dir, 'landmark/TrainVal1_fixed_class2/val')
    test_dir = os.path.join(dataset_dir, 'landmark/Test_Public')

num_training_samples=76500 #full dataset with
model_name='resnext50_32x4d'
input_sz=320

#hyper parameters

dataset='ZaloAILandmark'
classes = 103
batch_size=32
epochs=200
log_interval=200
num_workers=6
base_lr=0.001
lr_decay=0.75
lr_mode='step'
lr_decay_epoch='3,10,20,30,40,50,70,110,150,200,450,900,1500'
save_frequency=5

#training
resume_param=''
resume_state=''
resume_epoch=0

#testing
pretrained_param='resnext50_32x4d_320/ZaloAILandmark-resnext50_32x4d-20.params'
submission_prefix='30'

#data analyze
data_analyze_dir='data_analyze'
