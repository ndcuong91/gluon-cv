import matplotlib.pyplot as plt
import numpy as np
import os
import config_arm_project as config
import mxnet as mx
from mxboard import SummaryWriter
import scipy.spatial.distance as distance
import shutil, cv2
from gluoncv.model_zoo import get_model
from mxnet import image, init, nd, gluon, ndarray
from mxnet.gluon.data.vision import transforms

import utils_classification as utils
from utils_classification import get_list_file_in_folder, get_list_dir_in_folder, get_string_from_file
import test_arm_project as test

model_name=config.model_name
pretrained=''
num_class=config.classes
data_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal1/'
val_dir = config.val_dir
train_dir = config.train_dir
input_sz=config.input_sz
batch_size=config.batch_size
num_workers=config.num_workers

color_list=[]
color_list.append((0,0,1,1))
color_list.append((0,1,0,0.8))
color_list.append((1,0,0,0.8))
color_list.append((1,1,0,0.8))

resize_factor=1.5
jitter_param = 0.4
lighting_param = 0.1
mean_args = {'mean_r': 123.68, 'mean_g': 116.779, 'mean_b': 103.939}
std_args = {'std_r': 58.393, 'std_g': 57.12, 'std_b': 57.375}

def plot_bar(index, lists,labels, title=''):
    f, axes = plt.subplots(figsize=(20, 10), sharex=True)
    plt.xlabel('Class')
    plt.ylabel('Num of samples')
    plt.xticks(index, fontsize=8, rotation=90)
    plt.title(title)
    for i in range(len(lists)):
        plt.bar(index, lists[i],color=color_list[i],label=labels[i])
    plt.legend(loc='best')
    plt.savefig(os.path.join(data_analyze_dir, title.replace('/','_') +'_distribution.jpg'))
    plt.show()

def submission(net, test_time_augment=1, topk=3, use_tta_transform=False, submission_dir='submission'):
    print 'submission. Begin'
    name, topk_labels, topk_probs= test.classify_dir(net,test_dir,[mx.gpu()],topk=topk, test_time_augment=test_time_augment, use_tta_transform=use_tta_transform, sub_class=False)
    samples=name.shape[0]
    print 'data_dir:',test_dir,', num samples =',samples
    result='id,predicted\n'
    for i in range(samples):
        result +=str(name[i]) + ','
        for k in range(topk):
            if (k < 2):
                result += str(topk_labels[i][k]) + ' '
            else:
                result += str(topk_labels[i][k]) + '\n'

    submit_dir=os.path.join(submission_dir,submission_prefix)
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)
    submit_file=os.path.join(submit_dir,(os.path.splitext(pretrained_param)[0]).replace('/','_'))+'.csv'

    pretrained_param_name= os.path.basename(pretrained_param)
    shutil.copy(pretrained_param,os.path.join(submit_dir,pretrained_param_name))
    with open(submit_file, 'w') as file:
        file.write(result)
        print 'Save submission file to:',submit_file
    print 'Submission. Finish'


def get_result_from_dir(data_dir):
    list_image = []
    list_label=[]

    list_dir = get_list_dir_in_folder(data_dir)
    for dir in list_dir:
        imgs = get_list_file_in_folder(os.path.join(data_dir, dir))
        for img in imgs:
            name = img.split('_')
            length=len(name)
            list_image.append(name[length-1])
            list_label.append(dir.replace('_ok',''))
    return list_image, list_label

def submission_with_manual_result(net, submission_dir = 'submission'):
    print 'submission_with_manual_result. Begin'
    data_dir=os.path.join(config.dataset_dir,'Test_Public')
    dir=os.path.join(config.dataset_dir,'22_landmark_2405')

    img_1, label_1 = get_result_from_dir(os.path.join(dir,'22_Hand_classified'))
    img_2, label_2 = get_result_from_dir(os.path.join(dir,'22_Public_classified'))
    img_3, label_3 = get_result_from_dir(os.path.join(dir,'22_Public_classified_ok'))
    submit_key=list(zalo_list.submit.keys())

    name, topk_labels, topk_probs = test.classify_dir(net, data_dir, [mx.gpu()], test_time_augment=1, topk=3,
                                                 use_tta_transform=False, sub_class=False)

    for i in range(len(img_1)):
        idx = (np.where(name == int(img_1[i].replace('.jpg', ''))))[0][0]
        replace_result = True
        for k in range(3):
            if (topk_labels[idx][k] == int(label_1[i])):
                replace_result = False
        if (replace_result == True):
            topk_labels[idx][2] = int(label_1[i])

    for i in range(len(img_2)):
        idx = (np.where(name == int(img_2[i].replace('.jpg', ''))))[0][0]
        replace_result = True
        for k in range(3):
            if (topk_labels[idx][k] == int(label_2[i])):
                replace_result = False
        if (replace_result == True):
            topk_labels[idx][2] = int(label_2[i])


    for i in range(len(img_3)):
        idx = (np.where(name == int(img_3[i].replace('.jpg', ''))))[0][0]
        replace_result = True
        for k in range(3):
            if (topk_labels[idx][k] == int(label_3[i])):
                replace_result = False
        if (replace_result == True):
            topk_labels[idx][2] = int(label_3[i])

    samples = name.shape[0]
    print 'data_dir:', test_dir, ', num samples =', samples
    result = 'id,predicted\n'
    for i in range(samples):
        result += str(name[i]) + ','
        line = ''
        for key in submit_key:
            line = ''
            if(str(name[i])==key):
                line=zalo_list.submit[key]
                break
            else:
                for k in range(3):
                    if (k < 2):
                        line += str(topk_labels[i][k]) + ' '
                    else:
                        line += str(topk_labels[i][k]) + '\n'
        result+=line

    submit_dir = os.path.join(submission_dir, submission_prefix)
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)
    submit_file = os.path.join(submit_dir, (os.path.splitext(pretrained_param)[0]).replace('/', '_')) + '.csv'

    pretrained_param_name = os.path.basename(pretrained_param)
    shutil.copy(pretrained_param, os.path.join(submit_dir, pretrained_param_name))
    with open(submit_file, 'w') as file:
        file.write(result)
        print 'Save submission file to:', submit_file
    print 'Submission_with_manual_result. Finish'

def process_result(src_dir,des_dir, name, topk_labels, topk_probs):

    for i in range(num_class):
        if not os.path.exists(os.path.join(des_dir,str(i))):
            os.makedirs(os.path.join(des_dir,str(i)))

    num_samples=name.shape[0]
    for n in range(num_samples):
        src_path=os.path.join(src_dir,str(name[n])+'.jpg')
        dst_path=os.path.join(des_dir,str(topk_labels[n][0]),str((topk_probs[n][0]).round(5))+'_'+str(name[n])+'.jpg')
        shutil.copy(src_path,dst_path)
        kk=1


def draw_result(src_path, dst_path, topk_labels, topk_probs):
    origimg = cv2.imread(src_path)
    left = 5
    for k in range(5):
        top = 25 * (k + 1)
        title = "%d:%.4f,%s" % (topk_labels[k], topk_probs[k], label_strs[topk_labels[k]])
        width=len(title)
        cv2.rectangle(origimg, (left, top-20), (left+12*width, top+5), (200, 200, 200), -1)
        cv2.putText(origimg, title, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imwrite(dst_path, origimg)
    os.remove(src_path)

def draw_result_for_image():
    # public_classify_dir = '/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/22_landmark/22_Public_classified'
    # name, labels, topk_labels, topk_probs= classify_dir(finetune_net,public_classify_dir,[mx.gpu()],test_time_augment=1, topk=5, use_tta_transform=False)
    #
    # print public_classify_dir
    # for i in range(len(name)):
    #     file_name=str(name[i])+'.jpg'
    #     prob='%.4f'%topk_probs[i][0]
    #     new_name=prob+'_'+file_name
    #     full_path=os.path.join(public_classify_dir,str(labels[i]), file_name)
    #     new_path=os.path.join(public_classify_dir,str(labels[i]),new_name)
    #     draw_result(full_path,new_path, topk_labels[i], topk_probs[i])
    #
    #
    # hand_classify_dir = '/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/22_landmark/22_Hand_classified'
    #
    # for n in range(103):
    #     if not os.path.exists(os.path.join(hand_classify_dir,str(n))):
    #         os.makedirs(os.path.join(hand_classify_dir,str(n)))
    #
    # name, labels, topk_labels, topk_probs= classify_dir(finetune_net,hand_classify_dir,[mx.gpu()],test_time_augment=1, topk=5, use_tta_transform=False)
    #
    # print hand_classify_dir
    # for i in range(len(name)):
    #     file_name=str(name[i])+'.jpg'
    #     prob='%.4f'%topk_probs[i][0]
    #     new_name=prob+'_'+file_name
    #     full_path=os.path.join(hand_classify_dir,str(labels[i]), file_name)
    #     new_path=os.path.join(hand_classify_dir,str(labels[i]),new_name)
    #     draw_result(full_path,new_path, topk_labels[i], topk_probs[i])


    need_classify_dir = '/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/22_landmark/22_Need_classify2'
    name, topk_labels, topk_probs= test.classify_dir(finetune_net,need_classify_dir,[mx.gpu()],test_time_augment=1, topk=5, use_tta_transform=False, sub_class=False)

    print need_classify_dir
    for i in range(len(name)):
        file_name=str(name[i])+'.jpg'
        prob='%.4f'%topk_probs[i][0]
        new_name=prob+'_'+file_name
        full_path=os.path.join(need_classify_dir,file_name)
        new_path=os.path.join(need_classify_dir,new_name)
        draw_result(full_path,new_path, topk_labels[i], topk_probs[i])

def get_number_of_img_for_each_class_in_folder(data_dir):
    classes=get_list_dir_in_folder(data_dir)
    classes= sorted(classes, key=lambda x: int(x))
    num_imgs=[]
    for cls in classes:
        files=get_list_file_in_folder(os.path.join(data_dir,cls))
        num_imgs.append(len(files))
    return num_imgs


def plot_distribution_result(data_dir):
    lists=[]
    labels=[]
    index = np.arange(num_class)

    # num_samples = get_number_of_img_for_each_class_in_folder('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal_origin')
    # lists.append(num_samples)
    # labels.append('TrainVal1)

    file_path='data_analyze/public_test_top3_prob.txt'
    data1=map(int, get_string_from_file(file_path))
    lists.append(data1)
    labels.append('public_test_top3_prob')

    file_path='data_analyze/private_test_top3_prob.txt'
    data1=map(int, get_string_from_file(file_path))
    lists.append(data1)
    labels.append('private_test_top3_prob')

    plot_bar(index,lists,labels, title='public_test_top3_prob_vs_private_test_top3_prob')


def plot_distribution_of_images_in_folder(data_dir):
    num_samples=get_number_of_img_for_each_class_in_folder(data_dir)
    index = np.arange(len(num_samples))
    plot_bar(index,[num_samples])


def get_embedded_feature_and_draw_tSNE(net, ctx, data_dir, sub_class_inside=False,embedded_dir='data_analyze', icon=False, save_data=False, save_prefix=''):
    print 'get_embedded_feature. start'
    print 'data_dir:', data_dir

    if (sub_class_inside):
        total_name, total_label, total_outputs, total_resized_images = test.classify_dir_with_subclass(net, ctx, data_dir,use_tta_transform=False, seed=1, export_image=icon, soft_max=False)
    else:
        total_name, total_outputs, total_resized_images = test.classify_dir_wo_subclass(net, ctx, data_dir,use_tta_transform=False, seed=1, export_image=icon, soft_max=False)

    if(save_data):
        print 'Write embedded data to', embedded_dir
        mx.nd.save(os.path.join(embedded_dir, '%s_%s_embedding_feature.ndarray' % (save_prefix, model_name)),total_outputs)
        if (sub_class_inside):
            mx.nd.save(os.path.join(embedded_dir, '%s_%s_label.ndarray' % (save_prefix, model_name)), total_label)
        mx.nd.save(os.path.join(embedded_dir, '%s_%s_name.ndarray' % (save_prefix, model_name)), total_name)
        mx.nd.save(os.path.join(embedded_dir, '%s_%s_image_data.ndarray' % (save_prefix, model_name)),total_resized_images)



    with SummaryWriter(logdir='./logs') as sw:
        #sw.add_image(tag=model_name+'_images', image=total_resized_images)
        if (sub_class_inside):
            sw.add_embedding(tag=model_name+'_codes', embedding=total_outputs, labels=total_label, names=total_name, images=total_resized_images)
        else:
            sw.add_embedding(tag=model_name+'_codes', embedding=total_outputs, names=total_name, images=total_resized_images)

    print 'Call Mxboard'
    call_mxboard()

def call_mxboard():
    os.system('tensorboard --logdir=./logs --host=127.0.0.1 --port=8888')


def cluster_images_base_on_embedded_feature(model_name, key_img_file='', embedded_dir='data_analyze', save_prefix='', dis_thres=0.5):
    embedding_feature = mx.nd.load(os.path.join(embedded_dir,'%s_%s_embedding_feature.ndarray' % (save_prefix,model_name)))[0]
    names = mx.nd.load(os.path.join(embedded_dir,'%s_%s_name.ndarray' % (save_prefix,model_name)))[0].asnumpy()

    key_imgs=get_string_from_file(os.path.join(embedded_dir, key_img_file))
    num_samples=embedding_feature.shape[0]

    src_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public'
    des_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public_clustered'

    count=0
    img_count=0
    for img in key_imgs:
        print 'Process embedded feature of:',img,
        img_result=[]
        dis_result=[]
        idx_img= (np.where(names == int(img)))[0][0]
        embedded_vec=embedding_feature[idx_img]

        for i in range(num_samples):
            dis=distance_between_2_embedded_vector(embedded_vec, embedding_feature[i])
            if(dis<dis_thres):
                img_result.append(names[i])
                dis_result.append(dis)
                count+=1

        sorted_dis=np.sort(dis_result)
        sorted_idx=np.argsort(dis_result)
        sorted_name = [img_result[i] for i in sorted_idx]

        print ', similar embedded feature %d, threshold %.2f' % (len(dis_result),dis_thres)
        print 'Begin copy similar image to folder:',os.path.join(des_dir,'cluster'+'_'+str(img_count))
        copy_similar_image_to_dir(sorted_name, sorted_dis, src_dir, des_dir, des_folder_index=img_count, des_folder_prefix='cluster')
        img_count+=1

def copy_similar_image_to_dir(list_img, sorted_dis, src_dir, des_dir, des_folder_index, des_folder_prefix='cluster', ext='.jpg', maximum=450):
    des_dir=os.path.join(des_dir,des_folder_prefix+'_'+str(des_folder_index))
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    for i in range(min(len(list_img),maximum)):
        src_file=os.path.join(src_dir, str(list_img[i])+ext)
        des_file=os.path.join(des_dir, str(i)+'_'+str(sorted_dis[i].round(4))+'_'+str(list_img[i])+ext)
        shutil.copy(src_file,des_file)

def distance_between_2_embedded_vector(first_vec, second_vec, distance_type='cosine'):
    if(distance_type=='euclidean'):
        distance.euclidean(first_vec.asnumpy(),second_vec.asnumpy())
    if (distance_type == 'cosine'):
        dis= distance.cosine(first_vec.asnumpy(),second_vec.asnumpy())
    return dis

def get_list_clustered_img(clustered_dir):
    print 'get_list_clustered_img in folder:',clustered_dir
    list_clustered_class=get_list_dir_in_folder(clustered_dir)
    clustered_file=[]
    for cls in list_clustered_class:
        cls_dir=os.path.join(clustered_dir,cls)
        list_file=get_list_file_in_folder(cls_dir)
        for file in list_file:
            field=file.split('_')
            clustered_file.append(field[2])
    print 'There are',len(clustered_file),'files clustered.'
    return clustered_file

def copy_img_that_did_not_clustered(data_dir, clustered_file, des_dir):
    print 'copy_img_that_did_not_clustered from',data_dir,'to',des_dir
    list_file = get_list_file_in_folder(data_dir)

    for file in clustered_file:
        list_file.remove(file)

    for i in range(len(list_file)):
        if(i%100==0 and i>0):
            print i,' files copied'
        src_file=os.path.join(data_dir,list_file[i])
        dst_file=os.path.join(des_dir,list_file[i])
        shutil.copy(src_file,dst_file)

def get_list_images(data_dir):
    list_image=[]

    list_dir=get_list_dir_in_folder(data_dir)
    for dir in list_dir:
        imgs=get_list_file_in_folder(os.path.join(data_dir,dir))
        for img in imgs:
            name = img.split('_')
            length = len(name)
            list_image.append(name[length - 1])
    return list_image

def process_data(data_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public'):
    new_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/22_landmark/22_Need_classify2'

    list_public_test=get_list_file_in_folder(data_dir)
    list_public_classified=get_list_images('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/22_landmark/22_Public_classified')
    list_need_classified=get_list_file_in_folder('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/22_landmark/22_Need_classify')
    list_hand_classified=get_list_images('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/22_landmark/22_Hand_classified')

    for file in list_public_classified:
        list_public_test.remove(file)
    for file in list_need_classified:
        name = file.split('_')
        length = len(name)
        list_public_test.remove(name[length - 1])

    for file in list_hand_classified:
        print file
        list_public_test.remove(file)

    for file in list_public_test:
        print file
        shutil.copy(os.path.join(data_dir,file),os.path.join(new_dir,file))


if __name__ == "__main__":
    finetune_net = test.get_network_with_pretrained(config.model_name, 'arm_v4.4_180_9180.params')

    get_embedded_feature_and_draw_tSNE(finetune_net,[mx.gpu()], '/media/atsg/Data/datasets/gvh205_arm_project/new_dataset1_resize300_rename/test', sub_class_inside=True)
    #call_mxboard()
    #plot_distribution_result('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public_result')
    #cluster_images_base_on_embedded_feature(model_name, key_img_file='key_img_pubic1_test.txt', save_prefix='public1_test_imagenet')

    #process_data()

    # name, topk_labels, topk_probs = test.classify_dir(finetune_net, '/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/22_landmark_2605/22_Hand_classified', [mx.gpu()], test_time_augment=1, topk=3,
    #                                              use_tta_transform=False, sub_class=True)
    #test.cla
    #'/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/22_landmark_2605/22_Hand_classified'
    #submission_with_manual_result(finetune_net)

    #lst=get_list_clustered_img('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public_clustered')
    #copy_img_that_did_not_clustered('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public',lst,'/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public1')
    #plot_distribution_of_images_in_folder('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public_result')
    #plot_bar_x()

    #create_folder('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal_50samples')
    print 'Finish'