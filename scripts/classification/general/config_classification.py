
pc='duycuongAI'

train_dir='/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/TrainVal/train'
num_training_samples=79640
val_dir='/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/TrainVal/val'
test_dir='/home/duycuong/PycharmProjects/research/ZaloAIchallenge2018/landmark/Public'
model='resnet152_v2'
dataset='ZaloAILandmark'

#hyper parameters
resume_param='resnext50_32x4d_224/2019-05-02_18.26/ZaloAILandmark-resnext50_32x4d-best.params'
resume_state='resnext50_32x4d_224/2019-05-02_18.26/ZaloAILandmark-resnext50_32x4d-best.states'
resume_param=''
resume_state=''

batch_size=16
epochs=200
log_interval=200
classes = 103
input_sz=224
num_workers=6
lr_mode='step'
