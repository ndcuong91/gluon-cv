import torch, torchvision
import mxnet as mx
import caffe
import os
import cv2
import numpy as np
model_version='v4.4'
input_size=180
folder=os.path.join('model',model_version)
params_file = os.path.join(folder,'arm_v4.4_180_9101.params')
saved_folder=os.path.join(folder,'data')

caffe_proto=os.path.join(folder,'arm_'+model_version+'_'+str(input_size)+'.prototxt')
caffe_params=os.path.join(folder,'arm_'+model_version+'_'+str(input_size)+'.caffemodel')
caffe_net = caffe.Net(caffe_proto, caffe.TEST)

dataset='/home/atsg/PycharmProjects/gvh205/arm_project/to_customer/GVH205_ARM_project_training_environment/dataset/dataset3_resize300/test'
img_file='/media/atsg/Data/datasets/SUN_ARM_project/test/messy_dirty/4561.jpg'
label=['clean_normal','messy_dirty']

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
std=[255., 255., 255.]

if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)
weight_folder=saved_folder
bias_folder=saved_folder

params=dict()
params['layer']=['conv1','conv2','conv3','conv4','conv5','conv6','dense1','dense2','dense3']
params['conv1']=dict()
params['conv1']['name']='0'
params['conv1']['input_channel']=3
params['conv1']['input_size']=input_size
params['conv1']['output_channel']=16
params['conv1']['bn_name']='1'

params['conv2']=dict()
params['conv2']['name']='4'
params['conv2']['input_channel']=16
params['conv2']['input_size']=90
params['conv2']['output_channel']=32
params['conv2']['bn_name']='5'

params['conv3']=dict()
params['conv3']['name']='8'
params['conv3']['input_channel']=32
params['conv3']['input_size']=45
params['conv3']['output_channel']=64
params['conv3']['bn_name']='9'

params['conv4']=dict()
params['conv4']['name']='12'
params['conv4']['input_channel']=64
params['conv4']['input_size']=23
params['conv4']['output_channel']=128
params['conv4']['bn_name']='13'

params['conv5']=dict()
params['conv5']['name']='16'
params['conv5']['input_channel']=128
params['conv5']['input_size']=12
params['conv5']['output_channel']=128
params['conv5']['bn_name']='17'

params['conv6']=dict()
params['conv6']['name']='20'
params['conv6']['input_channel']=128
params['conv6']['input_size']=6
params['conv6']['output_channel']=256
params['conv6']['bn_name']='21'

params['dense1']=dict()
params['dense1']['name']='24'
params['dense1']['neural']='128'
params['dense2']=dict()
params['dense2']['name']='25'
params['dense2']['neural']='64'
params['dense3']=dict()
params['dense3']['name']='output'
params['dense3']['neural']='2'


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def write_file(data, filename, folder=saved_folder, txt=True):
    if not os.path.exists(folder):
        os.makedirs(folder)

    length=len(data.shape)
    total_value=1
    for i in range(length):
        total_value *= data.shape[i]

    data = data.reshape(total_value)
    if (txt == True):
        text = ''
        for i in range(total_value):
            text += str(data[i]) + '\n'

        with open(os.path.join(folder, filename), "w") as text_file:
            text_file.write(text)
    else:
        data.tofile(os.path.join(folder, filename))

def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    #
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
    #
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros( conv.weight.size(0) )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_( b_conv + b_bn )
    #
    # we're done
    return fusedconv

def test_fuse():
    torch.set_grad_enabled(False)
    x = torch.randn(16, 3, 256, 256)
    rn18 = torchvision.models.resnet18(pretrained=True)
    rn18.eval()
    net = torch.nn.Sequential(
        rn18.conv1,
        rn18.bn1
    )
    y1 = net.forward(x)
    fusedconv = fuse_conv_and_bn(net[0], net[1])
    y2 = fusedconv.forward(x)
    d = (y1 - y2).norm().div(y1.norm()).item()
    print("error: %.8f" % d)

def preprocess(img_path):
    src = cv2.imread(img_path)
    input_sz = (params['conv1']['input_size'], params['conv1']['input_size'])
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_sz)
    img = img.astype(np.float32)
    img -= mean
    img /= std
    img = img.transpose((2, 0, 1))
    return img

def test_forward(net, imgfile):
    img = preprocess(imgfile)

    net.blobs['data'].data[...] = img
    out = net.forward()

    pool2=net.blobs['pool2'].data[...]
    pool3=net.blobs['pool3'].data[...]
    pool4=net.blobs['pool4'].data[...]
    pool5=net.blobs['pool5'].data[...]
    conv6=net.blobs['conv6'].data[...]
    idx=np.argmax(out['prob'][0])
    #print label[idx]
    return label[idx]

def merge_bn_conv(save_caffe_model=True):
    torch.set_grad_enabled(False)
    mxnet_params = mx.ndarray.load(params_file)

    for layer in params['layer']:
        print(layer)
        name_weight = layer + '_weight'
        name_bias = layer + '_bias'
        if 'conv' in layer:
            conv_torch = torch.nn.Conv2d(params[layer]['input_channel'],
                                         params[layer]['output_channel'],
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1),
                                         bias=True)
            conv_torch.weight.copy_(torch.tensor(mxnet_params[params[layer]['name'] + '.weight'].asnumpy()))
            conv_torch.bias.copy_(torch.tensor(mxnet_params[params[layer]['name'] + '.bias'].asnumpy()))
            conv_torch.training = False

            bn_torch = torch.nn.BatchNorm2d(params[layer]['output_channel'])
            bn_torch.weight.copy_(torch.tensor(mxnet_params[params[layer]['bn_name'] + '.gamma'].asnumpy()))
            bn_torch.bias.copy_(torch.tensor(mxnet_params[params[layer]['bn_name'] + '.beta'].asnumpy()))
            bn_torch.running_mean = torch.tensor(mxnet_params[params[layer]['bn_name'] + '.running_mean'].asnumpy())
            bn_torch.running_var = torch.tensor(mxnet_params[params[layer]['bn_name'] + '.running_var'].asnumpy())
            bn_torch.training = False

            fusedconv = fuse_conv_and_bn(conv_torch, bn_torch)

            save_weight = fusedconv.weight.detach().numpy()
            write_file(save_weight, name_weight + '.bin', txt=False)
            write_file(save_weight, name_weight + '.txt')
            save_bias = fusedconv.bias.detach().numpy()
            write_file(save_bias, name_bias + '.bin', txt=False)
            write_file(save_bias, name_bias + '.txt')
            # save_bias.tofile(os.path.join(bias_folder,name_bias+'.bin'))

        else:  # dense
            save_weight = mxnet_params[params[layer]['name'] + '.weight'].asnumpy()
            write_file(save_weight, name_weight + '.bin', txt=False)
            write_file(save_weight, name_weight + '.txt')
            save_bias = mxnet_params[params[layer]['name'] + '.bias'].asnumpy()
            write_file(save_bias, name_bias + '.bin', txt=False)
            write_file(save_bias, name_bias + '.txt')

        caffe_net.params[layer][0].data[...] = save_weight
        caffe_net.params[layer][1].data[...] = save_bias

    if(save_caffe_model):
        caffe_net.save(caffe_params)
    print('Finish merge batchnorm to conv and generate data!')

def classify_dir(data_dir):
    caffe_net_with_pretrained=caffe.Net(caffe_proto, caffe_params, caffe.TEST)
    caffe.set_mode_gpu()
    samples = dict()
    samples[label[0]]=277
    samples[label[1]]=279

    true_pred=dict()
    true_pred[label[0]]=0
    true_pred[label[1]]=0

    classes=get_list_dir_in_folder(data_dir)

    for cls in classes:
        print cls
        list_files=get_list_file_in_folder(os.path.join(dataset,cls))
        for file in list_files:
            file_path=os.path.join(dataset,cls,file)
            pred=test_forward(caffe_net_with_pretrained, file_path)
            if(pred==cls):
                true_pred[cls]+=1
        accuracy=(float)(true_pred[cls])/(float)(samples[cls])
        print 'True pred:',true_pred[cls],',Total:',samples[cls],',Accuracy:',100*accuracy

    print 'Final Accuracy:',(float)(true_pred[label[0]]+true_pred[label[1]])/(float)(samples[label[0]]+samples[label[1]])

if __name__ == "__main__":
    #test_fuse()
    merge_bn_conv()
    classify_dir(dataset)

    print 'Finish'