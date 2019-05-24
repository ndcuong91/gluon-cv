import os

pc='duycuongAI'
pc='300'


if(pc=='duycuongAI'):
    dataset_dir='/media/duycuong/Data/Dataset/ZaloAIChallenge2018/landmark'
    train_dir = os.path.join(dataset_dir, 'TrainVal/train')
    val_dir = os.path.join(dataset_dir, '/TrainVal/val')
    test_dir = os.path.join(dataset_dir, 'Public')
else:
    dataset_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark'
    train_dir = os.path.join(dataset_dir, 'TrainVal1_fixed_class2_merged_22_landmark')
    val_dir = os.path.join(dataset_dir, 'TrainVal1_fixed_class2/val')
    test_dir = os.path.join(dataset_dir, 'landmark/Test_Public')

num_training_samples=100000 #full dataset with 100000
model_name='resnext50_32x4d'
input_sz=224

#hyper parameters

dataset='ZaloAILandmark'
classes = 103
batch_size=32
epochs=200
log_interval=200
num_workers=6
base_lr=0.00043
lr_decay=0.75
lr_mode='step'
lr_decay_epoch='10,20,30,50,80,110,150,200,450,900,1500'
save_frequency=5

#training
resume_param='ZaloAILandmark-resnext50_32x4d-20.params'
resume_state=''
resume_epoch=0

#testing
pretrained_param='ZaloAILandmark-resnext50_32x4d-20.params'
submission_prefix='34'

#data analyze
data_analyze_dir='data_analyze'
