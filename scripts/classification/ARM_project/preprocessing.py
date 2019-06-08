import cv2, os
import utils_classification as utils

train_path = '/media/atsg/Data/datasets/ImageNet/imagenet/train'
test_path = '/media/atsg/Data/datasets/ImageNet/imagenet/val'
dst_data='/home/atsg/.mxnet/datasets/imagenet/val'

base_w=200
base_h=200


def resize_image_in_dataset(data_dir):
    classes=utils.get_list_dir_in_folder(data_dir)
    count=0
    for cls in classes:
        print count,cls
        src_dir = os.path.join(data_dir,cls)
        list_img = utils.get_list_file_in_folder(src_dir, ext='JPEG')
        dst_dir = os.path.join(dst_data,cls)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img_name in list_img:
            #print img_name
            src_path = os.path.join(src_dir,img_name)
            dst_path = os.path.join(dst_dir,img_name)
            origin = cv2.imread(src_path)
            if (origin.shape[0]*origin.shape[1]<40001):
                cv2.imwrite(dst_path, origin)
                continue
            resized = cv2.resize(origin, (base_w,base_h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(dst_path,resized)
        count+=1

resize_image_in_dataset(test_path)

