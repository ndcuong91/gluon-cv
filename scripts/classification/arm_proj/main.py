import mxnet as mx
import cv2
import time
import arm_v4_4 as arm_network
import os
import numpy as np

ctx = mx.cpu()
shape = 180
class_names = ["clean", "messy"]

#modified params
write_file=False #for test
image_file='/home/atsg/PycharmProjects/gvh205/others/images/dog.jpg'
quantize=False

model_version='v4.4'
csv_file = os.path.join('model',model_version, 'arm_' + model_version +'_'+str(shape)+ '_9101_9119.csv')
print csv_file
data_folder=os.path.join('model',model_version, 'data')  #for weight and bias
label=['clean_normal','messy_dirty']
multi=True
input_folder = 'to_customer/GVH205_ARM_project_training_environment/dataset/dataset3_resize300/test'

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
std=[255., 255., 255.]

def load_image(image_file):
    origimg = cv2.imread(image_file)
    img = cv2.resize(origimg, (shape, shape))

    img = np.array(img) - np.array([123.675, 116.28, 103.53])
    #img = np.array(img) / np.array([58.395, 57.12, 57.375])
    img = np.array(img) / np.array([255, 255, 255])

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    x = img[np.newaxis, :]
    return x

def preprocess(img_path):
    src = cv2.imread(img_path)
    input_sz = (shape, shape)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_sz)
    img = img.astype(np.float32)
    img -= mean
    img /= std
    img = img.transpose((2, 0, 1))
    x = img[np.newaxis, :]
    return x

# def display(img, out):
#     import random
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#     mpl.rcParams['figure.figsize'] = (10,10)
#     pens = dict()
#     plt.clf()
#     plt.imshow(img)
#     count=0
#     for det in out:
#         cid = int(det[0])
#         if cid < 0:
#             continue
#         score = det[1]
#         if score < thresh:
#             continue
#         count += 1
#         if cid not in pens:
#             pens[cid] = (random.random(), random.random(), random.random())
#         scales = [img.shape[1], img.shape[0]] * 2
#         xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
#         rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
#                              edgecolor=pens[cid], linewidth=3)
#         plt.gca().add_patch(rect)
#         text = class_names[cid]
#         plt.gca().text(xmin, ymin-2, '{:s} {:.3f}'.format(text, score),
#                        bbox=dict(facecolor=pens[cid], alpha=0.5),
#                        fontsize=12, color='white')
#         #print(str(text)+", Score: "+str(score))
#     print("total object detected: "+str(count))
#     plt.show()

def isImage(filename):
    isImg = filename.endswith('.png') or filename.endswith('.jpg')
    return isImg

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def detect_single(model, img_path, quantize=False, test=False, no_bias=False):
    if(test==False):
        output = model.classify(load_image(img_path),img_path, quantize=quantize, no_bias=no_bias)
    else:
        output=0

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #display(image, output.asnumpy()[0])
    return

def detect_multi(model,img_file, quantize=False, no_bias=False):

    samples = dict()
    samples[label[0]] = 277
    samples[label[1]] = 279

    true_pred = dict()
    true_pred[label[0]] = 0
    true_pred[label[1]] = 0

    classes = get_list_dir_in_folder(input_folder)

    for cls in classes:
        print cls
        list_files = get_list_file_in_folder(os.path.join(input_folder, cls))
        list_files=sorted(list_files)
        count=0
        for file in list_files:
            if(count%50==0 and count>0):
                print 'Process:',count,'files'
            #print file
            file_path = os.path.join(input_folder, cls, file)
            pred = model.classify(preprocess(file_path), file.replace('.jpg', ''), quantize=quantize, no_bias=no_bias)
            idx = np.argmax(pred.asnumpy())
            #test_forward(caffe_net_with_pretrained, file_path)
            if (label[idx] == cls):
                true_pred[cls] += 1
            count+=1
        accuracy = (float)(true_pred[cls]) / (float)(samples[cls])
        print 'True pred:', true_pred[cls], ',Total:', samples[cls], ',Accuracy:', 100 * accuracy

    print 'Final Accuracy:', (float)(true_pred[label[0]] + true_pred[label[1]]) / (float)(
        samples[label[0]] + samples[label[1]])

arm = arm_network.Model(model_version, csv_file, data_folder,data_folder,quantize=quantize, writefile=write_file)
begin = time.time()

if(multi==True):
    detect_multi(arm,image_file, quantize=quantize, no_bias=False)
else:
    detect_single(arm,image_file,quantize=quantize, no_bias=False)

print("processing time: "+str(time.time()-begin))
print('end')

