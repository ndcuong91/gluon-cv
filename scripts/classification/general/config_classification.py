
pc='duycuongAI'
pc='300'

if(pc=='duycuongAI'):
    train_dir='/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/TrainVal/train'
    val_dir='/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/TrainVal/val'
    test_dir='/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/Public'
else:
    train_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal_origin'
    val_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal1/val'
    test_dir = '/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public'

num_training_samples=86505 #full dataset with
model_name='resnext50_32x4d'
input_sz=320

#hyper parameters

dataset='ZaloAILandmark'
classes = 103
batch_size=16
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
pretrained_param='resnext50_32x4d_320/2019-05-10_15.39/ZaloAILandmark-resnext50_32x4d-best.params'
submission_file='16_submission_CuongND_resnet152_v2_224_val_acc_992.csv'
