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
model_name= 'arm_network_v4.4.1'

#hyper parameters
batch_size=128
epochs=120
log_interval=200
num_workers=8

training=True
resume_param = ''
resume_state = ''
resume_epoch = 0

if (model_name=='arm_network_v3.4'):
    input_sz=112
    resume_param = 'pretrained/ImageNet_v3.4_3452.params'
    resume_param = ''
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
if (model_name=='arm_network_v4.4' or model_name=='arm_network_v4.4.1'):
    input_sz=180
    resume_param = 'pretrained/ImageNet_v4.4_4000_180.params'

if(training==True):
    base_lr=0.4
    lr_decay=0.1
    lr_mode='cosine'
    lr_decay_epoch='40,60'
else: #finetuning
    base_lr=0.0001
    lr_decay=0.5
    lr_mode='step'
    lr_decay_epoch='500,1000'
    resume_param = 'arm_network_v4.5.3_180_8758/2019-06-01_22.10/ARM_New_Dataset-arm_network_v4.5.3-best-1072.params'
    resume_state = ''
    resume_epoch = 0

save_frequency=50
