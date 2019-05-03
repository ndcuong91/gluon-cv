from train_srgan import SRGenerator
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import os
import cv2
from mxnet import image
import argparse

folder='LR_test'
img_name='1.jpg,2.jpg,3.jpg,4.jpg,5.jpg,6.jpg,7.jpg'
epoch_snapshot='70'
pretrained='/media/atsg/Data/CuongND/srgan/snapshot_face_FLW/netG_epoch_'+epoch_snapshot+'.params'

def parse_args():
    parser = argparse.ArgumentParser(description='Test with srgan gan networks.')
    parser.add_argument('--images', type=str, default=img_name,
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpu_id', type=str, default='-1',
                        help='gpu id: e.g. 0. use -1 for CPU')
    parser.add_argument('--pretrained', type=str, default=pretrained,
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = parse_args()
    # context list
    if opt.gpu_id == '-1':
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(int(opt.gpu_id.strip()))

    netG = SRGenerator()
    netG.load_parameters(opt.pretrained)
    netG.collect_params().reset_ctx(ctx)
    image_list = [x.strip() for x in opt.images.split(',') if x.strip()]
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ax = None
    for image_name in image_list:
        print(image_name)
        image_path=os.path.join(folder,image_name)
        img = image.imread(image_path)
        img = transform_fn(img)
        img = img.expand_dims(0).as_in_context(ctx)
        output = netG(img)
        predict = mx.nd.squeeze(output)
        predict = ((predict.transpose([1,2,0]).asnumpy() * 0.5 + 0.5) * 255).astype('uint8')
        final = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(folder,'recover_'+epoch_snapshot+'_'+image_name),final)
        #cv2.imshow('result',final)
        #k = cv2.waitKey(0) & 0xff
        # Exit if ESC pressed
