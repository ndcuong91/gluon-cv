
pc='duycuongAI'
pc='300'

if(pc=='duycuongAI'):
    train_dir='/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/TrainVal/train'
    val_dir='/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/TrainVal/val'
    test_dir='/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/Public'
else:
    train_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal1/train'
    val_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal1/val'
    test_dir = '/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public'

num_training_samples=79640
model_name='resnext50_32x4d'
input_sz=224

#hyper parameters

dataset='ZaloAILandmark'
classes = 103
batch_size=16
epochs=200
log_interval=200
num_workers=6
lr_mode='step'
resume_param='' #'resnext50_32x4d_224/2019-05-02_18.26/ZaloAILandmark-resnext50_32x4d-best.params'
resume_state='' #'resnext50_32x4d_224/2019-05-02_18.26/ZaloAILandmark-resnext50_32x4d-best.states'
resume_epoch=0
