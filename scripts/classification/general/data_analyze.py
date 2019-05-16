import matplotlib.pyplot as plt
import numpy as np
import os
import config_classification as config
import mxnet as mx
from mxboard import SummaryWriter
import scipy.spatial.distance as distance
import shutil

model_name=config.model_name
pretrained=''
num_class=config.classes
data_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal1/'
val_dir = config.val_dir
train_dir = config.train_dir
data_analyze_dir = config.data_analyze_dir

color_list=[]
color_list.append((0,0,1,1))
color_list.append((0,1,0,0.8))
color_list.append((1,0,0,0.8))
color_list.append((1,1,0,0.8))

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

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

def get_number_of_img_for_each_class_in_folder(data_dir):
    classes=get_list_dir_in_folder(data_dir)
    classes= sorted(classes, key=lambda x: int(x))
    num_imgs=[]
    for cls in classes:
        files=get_list_file_in_folder(os.path.join(data_dir,cls))
        num_imgs.append(len(files))
    return num_imgs

def get_data_from_file(file_path):
    data = [line.rstrip('\n') for line in open(file_path)]
    # with open(file_path) as f:
    #     data = f.readlines()
    return data

def plot_distribution_result(data_dir):
    lists=[]
    labels=[]
    index = np.arange(num_class)

    # num_samples = get_number_of_img_for_each_class_in_folder(data_dir)
    # lists.append(num_samples)
    # labels.append('TrainVal_origin')

    # num_samples = get_number_of_img_for_each_class_in_folder('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal_origin')
    # lists.append(num_samples)
    # labels.append('TrainVal1)

    file_path='data_analyze/public_test_top3_prob.txt'
    data1=map(int, get_data_from_file(file_path))
    lists.append(data1)
    labels.append('public_test_top3_prob')


    file_path='data_analyze/private_test_top3_prob.txt'
    data1=map(int, get_data_from_file(file_path))
    lists.append(data1)
    labels.append('private_test_top3_prob')

    #get data from file
    # file_path='result_val_true_pred_top5.txt'
    # data1=map(int, get_data_from_file(file_path))
    # lists.append(data1)
    # labels.append(file_path.replace('result_','').replace('.txt',''))
    #   get data from file
    # file_path='result_val_true_pred_top1.txt'
    # data1=map(int, get_data_from_file(file_path))
    # lists.append(data1)
    # labels.append(file_path.replace('result_','').replace('.txt',''))

    plot_bar(index,lists,labels, title='public_test_top3_prob_vs_private_test_top3_prob')


def plot_distribution_of_images_in_folder(data_dir):
    num_samples=get_number_of_img_for_each_class_in_folder(data_dir)
    index = np.arange(len(num_samples))
    plot_bar(index,[num_samples])

def represent_tSNE_of_embedded_feature(model_name, embedded_dir='data_analyze', save_prefix=''):
    embedding_feature = mx.nd.load(os.path.join(embedded_dir,'%s_%s_embedding_feature.ndarray' % (save_prefix,model_name)))[0]
    names = mx.nd.load(os.path.join(embedded_dir,'%s_%s_name.ndarray' % (save_prefix,model_name)))[0].asnumpy()

    with SummaryWriter(logdir='./logs') as sw:
        sw.add_embedding(tag=model_name+'_codes', embedding=embedding_feature, labels=names)

def create_folder(dir, num_class=103):
    for i in range(num_class):
        class_dir=os.path.join(dir,str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def cluster_images_base_on_embedded_feature(model_name, key_img_file='', embedded_dir='data_analyze', save_prefix='', dis_thres=0.5):
    embedding_feature = mx.nd.load(os.path.join(embedded_dir,'%s_%s_embedding_feature.ndarray' % (save_prefix,model_name)))[0]
    names = mx.nd.load(os.path.join(embedded_dir,'%s_%s_name.ndarray' % (save_prefix,model_name)))[0].asnumpy()

    key_imgs=get_data_from_file(os.path.join(embedded_dir, key_img_file))
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



if __name__ == "__main__":
    #plot_distribution_result('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public_result')
    #plot_distribution_result('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal_origin')

    represent_tSNE_of_embedded_feature(model_name,save_prefix='trainval1_train_2_imagenet')
    #cluster_images_base_on_embedded_feature(model_name, key_img_file='key_img_pubic1_test.txt', save_prefix='public1_test_imagenet')

    #lst=get_list_clustered_img('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public_clustered')
    #copy_img_that_did_not_clustered('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public',lst,'/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public1')
    #plot_distribution_of_images_in_folder('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public_result')
    #plot_bar_x()

    #create_folder('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal_50samples')
    print 'Finish'