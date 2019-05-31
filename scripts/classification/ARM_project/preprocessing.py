import cv2, os
import config_ARM_project as config
import utils_classification as utils

train_path = config.train_dir
test_path = config.val_dir
dst_data='/home/atsg/PycharmProjects/gvh205/ARM_New_Dataset_Resize/test'

base_w=1000
base_h=1000


def resize_image_in_dataset(data_dir):
    classes=utils.get_list_dir_in_folder(data_dir)

    for cls in classes:
        print cls
        src_dir= os.path.join(data_dir,cls)
        list_img=utils.get_list_file_in_folder(src_dir)
        dst_dir=os.path.join(dst_data,cls)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img_name in list_img:
            print img_name
            src_path=os.path.join(src_dir,img_name)
            dst_path=os.path.join(dst_dir,img_name)
            img=cv2.imread(src_path)
            resized = cv2.resize(img, (base_w,base_h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(dst_path,resized)
            kk=1

resize_image_in_dataset(test_path)

