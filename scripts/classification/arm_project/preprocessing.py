import cv2, os
import utils_classification as utils

data_dir='/media/atsg/Data/datasets/gvh205_arm_project/new_dataset1_resize300'
src_train =os.path.join(data_dir,'train')
dst_train=os.path.join(data_dir,'train_new')
src_test = os.path.join(data_dir,'test')
dst_test= os.path.join(data_dir,'test_new')

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
        list_img=sorted(list_img)
        print len(list_img)
        dst_dir = os.path.join(dst_path,cls)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        rename=0
        result=''
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

def rename_and_resize_img_in_dataset(src_path, dst_path):
    classes=utils.get_list_dir_in_folder(src_path)
    classes=sorted(classes)
    count=0
    rename=0
    for cls in classes:
        src_dir = os.path.join(src_path,cls)
        list_img = utils.get_list_file_in_folder(src_dir, ext='jpg')
        list_img=sorted(list_img)
        print count,cls, len(list_img)
        dst_dir = os.path.join(dst_path,cls)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        result=''
        for img_name in list_img:
            print rename,img_name
            result+=str(rename)+' '+img_name.replace('.jpg','')+'\n'
            src_img = os.path.join(src_dir,img_name)
            dst_img = os.path.join(dst_dir,str(rename)+'.jpg')
            origin = cv2.imread(src_img)
            resized = cv2.resize(origin, (base_w,base_h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(dst_img,resized)
            rename+=1

        with open(cls+'.txt', 'w') as f:
            f.write(result)
        count+=1


rename_and_resize_img_in_dataset(src_train, dst_train)
#rename_and_resize_img_in_dataset(src_test, dst_test)