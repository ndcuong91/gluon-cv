"""SSD Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt

pretrained='train_mobilenetSSD_same_as_Caffe_parse_params_7337/ssd_300_mobilenet1.0_voc_same_as_caffe_7337.params'
shape=300
def parse_args():
    parser = argparse.ArgumentParser(description='Test with SSD networks.')
    parser.add_argument('--network', type=str, default='ssd_300_mobilenet1.0_voc',
                        help="Base network name")
    parser.add_argument('--images', type=str, default='/home/atsg/PycharmProjects/gvh205/others/dog.jpg',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default=pretrained,
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    # grab some image if not specified
    image_list = [x.strip() for x in args.images.split(',') if x.strip()]

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(args.network, pretrained=False, pretrained_base=False)
        net.load_parameters(args.pretrained)
    net.set_nms(0.45, 200)
    net.collect_params().reset_ctx(ctx = ctx)

    for image in image_list:
        ax = None
        x, img = presets.ssd.load_test(image, short=shape)
        x = x.as_in_context(ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                                    class_names=net.classes, ax=ax)
        plt.show()
