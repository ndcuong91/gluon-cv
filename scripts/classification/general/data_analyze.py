import matplotlib.pyplot as plt
import numpy as np
import os

num_class=103
data_dir='/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/TrainVal1/'
val_dir = os.path.join(data_dir, 'val')
train_dir = os.path.join(data_dir, 'train')

color_list=[]
color_list.append((1,0,0,0.8))
color_list.append((0,1,0,0.8))
color_list.append((0,0,1,0.8))
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

    file_path='result_public_test_top3_prob.txt'
    data1=map(int, get_data_from_file(file_path))
    lists.append(data1)
    labels.append(file_path.replace('result_','').replace('.txt',''))

    num_samples = get_number_of_img_for_each_class_in_folder(data_dir)
    lists.append(num_samples)
    labels.append('ground_truth')

    #get data from file
    # file_path='result_val_true_pred_top5.txt'
    # data1=map(int, get_data_from_file(file_path))
    # lists.append(data1)
    # labels.append(file_path.replace('result_','').replace('.txt',''))

    #get data from file

    #get data from file
    # file_path='result_val_true_pred_top1.txt'
    # data1=map(int, get_data_from_file(file_path))
    # lists.append(data1)
    # labels.append(file_path.replace('result_','').replace('.txt',''))

    plot_bar(index,lists,labels)


def plot_distribution_of_images_in_folder(data_dir):
    num_samples=get_number_of_img_for_each_class_in_folder(data_dir)
    index = np.arange(len(num_samples))
    plot_bar(index,[num_samples])

if __name__ == "__main__":
    #plot_distribution_result('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public_result')
    plot_distribution_result(val_dir)
    #plot_distribution_of_images_in_folder('/media/atsg/Data/datasets/ZaloAIChallenge2018/landmark/Test_Public_result')
    #plot_bar_x()