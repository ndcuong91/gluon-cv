import cv2, os
import utils_classification as utils

src_train = '/media/atsg/Data/datasets/ImageNet/imagenet/train'
dst_train='/home/atsg/PycharmProjects/gvh205/to_customer/training_arm_project/dataset/dataset3_resize300/train'
#src_test = '/media/atsg/Data/datasets/SUN_ARM_project/test'
#dst_test='/home/atsg/PycharmProjects/gvh205/to_customer/training_arm_project/dataset/dataset3_resize300/test'

base_w=300
base_h=300


def resize_image_in_dataset(src_path, dst_path):
    classes=utils.get_list_dir_in_folder(src_path)
    classes=sorted(classes)
    count=0
    for cls in classes:
        print count,cls,
        src_dir = os.path.join(src_path,cls)
        list_img = utils.get_list_file_in_folder(src_dir, ext='JPEG')
        print len(list_img)
        count+=1
        continue
        dst_dir = os.path.join(dst_path,cls)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img_name in list_img:
            print img_name
            src_img = os.path.join(src_dir,img_name)
            dst_img = os.path.join(dst_dir,img_name)
            origin = cv2.imread(src_img)
            if (origin.shape[0]*origin.shape[1]<90001):
                cv2.imwrite(dst_img, origin)
                continue
            resized = cv2.resize(origin, (base_w,base_h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(dst_img,resized)
        count+=1

resize_image_in_dataset(src_train, dst_train)
#resize_image_in_dataset(src_test, dst_test)