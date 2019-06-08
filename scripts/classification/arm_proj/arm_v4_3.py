from mxnet import nd
import mxnet as mx
import cv2
import numpy as np
import tvm
import math
import pandas as pd
from ast import literal_eval
#from joblib import Parallel, delayed
import os
import time

class Model:
    #model_file: .csv file that contain detail information abt model, e.g mobilenet_SSD_300
    def __init__(self, model_version, model_file, weight_folder, bias_folder='', quantize=False, writefile=False):
        self.model_version=model_version
        self.writefile=writefile
        self.layer_names=[]
        self.input_layer=[]
        self.weight_shapes = []
        self.bias_shapes = []
        self.output_shapes = []
        self.kernel = []
        self.stride = []
        self.pad = []
        self.relu=[]
        self.weight_scales = []
        self.data_scales = []
        self.data_scales_prev = []
        self.no_bias=False

        self.input_data = []
        self.output_data = []
        self.weights = []
        self.biases = []

        data = pd.read_csv(model_file)
        self.total_layer = 12
        self.total_column = data.shape[1]

        self.append_value(data,'layer_name',self.layer_names)
        self.append_value(data,'input_layer',self.input_layer)
        self.append_value(data,'weight_shape',self.weight_shapes,tuple=True)
        self.append_value(data,'bias_shape',self.bias_shapes,tuple=True)
        self.append_value(data,'output_shape',self.output_shapes,tuple=True)
        self.append_value(data,'kernel',self.kernel,tuple=True)
        self.append_value(data,'stride',self.stride,tuple=True)
        self.append_value(data,'pad',self.pad,tuple=True)
        self.append_value(data,'relu',self.relu)

        if(quantize==True):
            self.append_value(data, 'weight_scale', self.weight_scales, multiple=True)
            self.append_value(data, 'data_scale', self.data_scales)
            self.append_value(data, 'data_scale_prev', self.data_scales_prev)

        self.get_weight(weight_folder)
        if(bias_folder==''):
            print('No bias')
            self.no_bias==True
        else:
            self.get_bias(bias_folder)
        print('Arm_v4_3.py. Finish initialize parameters.')

    def write_file(self, data, filename, folder='data_demo', bias=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        if(bias==False):
            #data = data.transpose((0, 2, 3, 1))
            total_value=data.shape[1] * data.shape[2] * data.shape[3] * data.shape[0]
        else:
            total_value=data.shape[0]

        data=data.reshape(total_value)
        text=''
        for i in range (total_value):
            text+=str(data[i])+'\n'

        with open(os.path.join(folder,filename), "w") as text_file:
            text_file.write(text)

    def write_file_output(self, data,filename, folder='data_demo', txt=True): #txt or bin file
        if not os.path.exists(folder):
            os.makedirs(folder)
        #data = data.transpose((0, 2, 3, 1))
        if(txt==True):
            total_value=data.shape[1] * data.shape[2] * data.shape[3] * data.shape[0]
            data=data.reshape(total_value)
            text=''
            for i in range (total_value):
                text+=str(data[i])+'\n'

            with open(os.path.join(folder,filename), "w") as text_file:
                text_file.write(text)
        else:
            data.tofile(os.path.join(folder,filename))


    def read_data(self, filename, len_to_read=-1):  # read .bin or .txt data file

        if ('.txt' in filename):
            with open(filename) as f:
                lines = []
                if (len_to_read > -1):
                    count = 0
                    for line in f:
                        if (count < len_to_read):
                            lines.append(line)
                        count = count + 1
                else:
                    lines = f.readlines()
                x = np.array(lines)
                data = x.astype(np.float)

        if ('.bin' in filename):
            data = np.fromfile(os.path.join(filename), dtype='float32')

        return data

    def append_value(self, data, feature, array, boole=False, tuple=False, multiple=False):
        if(multiple==False):
            column = data[feature]
            for i in range(self.total_layer):
                if(tuple==True):
                    if(pd.isnull(column.iloc[i])):
                        array.append(-1)
                    else:
                        array.append(literal_eval(column.iloc[i]))
                else:
                    if(pd.isnull(column.iloc[i])):
                        array.append(-1)
                    else:
                        array.append(column.iloc[i])
        else:
            x = data.columns.get_loc(feature)
            for i in range(self.total_layer):
                values = []
                if('pool' not in self.layer_names[i]):
                    for n in range(self.bias_shapes[i][0]):
                        values.append(data.iat[i, x + n])
                    array.append(values)
                else:
                    array.append(-1)



        #convert .txt to bin
        # import caffe
        # net = caffe.Net('/home/prdcv/PycharmProjects/gvh205/MobileNet-SSD/no_bn.prototxt', '/home/prdcv/PycharmProjects/gvh205/MobileNet-SSD/no_bn.caffemodel', caffe.TEST)
        # for i in range(self.total_layer):
        #     layer_name=self.layer_names[i]
        #     W = net.params[layer_name][0].data[...]
        #     b = net.params[layer_name][1].data[...]
        #
        #     W.tofile('/home/prdcv/PycharmProjects/gvh205/Pruning/caffe_model/original_channel_quantize/'+ self.layer_names[i].replace('/dw', '_dw') + '_weight.bin')
        #     b.tofile('/home/prdcv/PycharmProjects/gvh205/Pruning/caffe_model/original_channel_quantize/'+ self.layer_names[i].replace('/dw', '_dw') + '_bias.bin')
        #     values = []

    def get_weight(self, weight_folder):
        for i in range(self.total_layer):
            layer_name=self.layer_names[i].replace('/','_')
            print('get weight in ' +layer_name)
            weight = np.fromfile(os.path.join(weight_folder, layer_name + '_weight.bin'), dtype='float32')
            weight=np.reshape(weight,self.weight_shapes[i])
            #weight=weight.transpose((0, 2, 3, 1))
            self.weights.append(np.reshape(weight,self.weight_shapes[i]))

    def get_bias(self, bias_folder):
        for i in range(self.total_layer):
            layer_name=self.layer_names[i].replace('/','_')
            print('get bias in ' +layer_name)
            bias = np.fromfile(os.path.join(bias_folder,layer_name+ '_bias.bin'), dtype='float32')
            self.biases.append(np.reshape(bias,self.bias_shapes[i]))

    def conv_normal(self,id,i, input_data,output_data,sf_in, kernel, pad, stride):
        #print('conv_normal '+str(i))
        if (self.weight_scales[id][i] == 0):
            return
        scaling_factor_i = self.weight_scales[id][i] / 127
        weight_shape = self.weights[id][i].shape
        weight_channel_i = self.weights[id][i].reshape(1, weight_shape[0], weight_shape[1], weight_shape[2])
        weight_int8_i = np.clip((weight_channel_i / scaling_factor_i).round(), -127, 127)

        top_blob_int32_i = nd.Convolution(data=nd.array(input_data), weight=nd.array(weight_int8_i),
                                          bias=None, kernel=kernel, pad=pad, stride=stride,
                                          num_filter=1, no_bias=True, num_group=1)

        # dequantize
        conv_i = top_blob_int32_i * sf_in * scaling_factor_i
        output_data[0][i] = conv_i[0][0] + self.biases[id][i]


    def convolution(self, input, id, no_bias=False, quantize=False):
        filter=self.bias_shapes[id][0]
        kernel=self.kernel[id]
        pad=self.pad[id]
        stride=self.stride[id]
        layer_name=self.layer_names[id].replace('/','_')
        if (self.is_depthwise[id] == 1):
            group =filter
        else:
            group = 1

        if (quantize == True):
            if (self.is_depthwise[id] == False):
                sf_in = self.data_scales_prev[id] / 127

                if (self.writefile):
                    sf_out = self.data_scales[id] / 127
                    bias_qt8 = nd.zeros(self.bias_shapes[id])
                    scale_rate_fp32 = nd.zeros(self.bias_shapes[id])
                    weight_qt8 = nd.zeros(self.weight_shapes[id])
                data_int8 = np.clip((input / sf_in).round(), -127, 127)

                conv = nd.zeros(self.output_shapes[id])

                # sequential conv
                for i in range(self.bias_shapes[id][0]):
                    if (self.writefile):
                        b_i = nd.array(self.biases[id][i].reshape(1, ))
                        b_i = np.clip((b_i / sf_out).round(), -127, 127)
                        bias_qt8[i] = b_i
                    if (self.weight_scales[id][i] == 0):
                        conv[0][i] = self.biases[id][i]
                        continue
                    scaling_factor_i = self.weight_scales[id][i] / 127
                    weight_shape = self.weights[id][i].shape
                    weight_channel_i = self.weights[id][i].reshape(1, weight_shape[0], weight_shape[1], weight_shape[2])
                    weight_int8_i = np.clip((weight_channel_i / scaling_factor_i).round(), -127, 127)

                    top_blob_int32_i = nd.Convolution(data=nd.array(data_int8), weight=nd.array(weight_int8_i),
                                                      bias=None, kernel=kernel, pad=pad, stride=stride,
                                                      num_filter=1, no_bias=True, num_group=group)

                    # dequantize
                    if (self.writefile):
                        scale_rate_fp32[i] = sf_out
                        weight_qt8[i] = weight_int8_i
                    conv_i = top_blob_int32_i * sf_in * scaling_factor_i
                    conv[0][i] = conv_i[0][0] + self.biases[id][i]

                # non-depthwise
                if (self.writefile):
                    self.write_file_output(input, 'input_' + layer_name + '.txt')
                    self.write_file_output(data_int8, 'input_qt8_' + layer_name + '.txt')
                    # self.write_file(self.weights[id], 'weight_' + layer_name + '.txt')
                    # self.write_file(self.biases[id], 'bias_' + layer_name + '.txt', bias=True)
                    # self.write_file(weight_qt8.asnumpy(), 'weight_qt8_' + layer_name + '.txt')
                    # self.write_file(bias_qt8.asnumpy(), 'bias_qt8_' + layer_name + '.txt', bias=True)
                    # self.write_file(scale_rate_fp32.asnumpy(), 'sf_out_fp32_' + layer_name + '.txt', bias=True)
                    # self.write_file(scale_rate_fp32.asnumpy(), 'scale_rate_fp32_' + layer_name + '.txt', bias=True)
            else:
                sf_in = self.data_scales_prev[id] / 127

                if (self.writefile):
                    sf_out = self.data_scales[id] / 127
                    bias_qt8 = nd.zeros(self.bias_shapes[id])
                    scale_rate_fp32 = nd.zeros(self.bias_shapes[id])
                    weight_qt8 = nd.zeros(self.weight_shapes[id])

                data_int8 = np.clip((input / sf_in).round(), -127, 127)

                conv = nd.zeros(self.output_shapes[id])

                for i in range(self.bias_shapes[id][0]):
                    if (self.writefile):
                        b_i = nd.array(self.biases[id][i].reshape(1, ))
                        b_i = np.clip((b_i / sf_out).round(), -127, 127)
                        bias_qt8[i] = b_i
                    if (self.weight_scales[id][i] == 0):
                        conv[0][i] = self.biases[id][i]
                        continue
                    scaling_factor_i = self.weight_scales[id][i] / 127
                    weight_shape = self.weights[id][i].shape
                    weight_channel_i = self.weights[id][i].reshape(1, weight_shape[0], weight_shape[1], weight_shape[2])
                    weight_int8_i = np.clip((weight_channel_i / scaling_factor_i).round(), -127, 127)

                    if (id == 0):
                        weight_int8_i = np.flip(weight_int8_i, 1)

                    data_shape = data_int8[0][i].shape
                    data_int8_i = data_int8[0][i].reshape(1, 1, data_shape[0], data_shape[1])
                    top_blob_int32_i = nd.Convolution(data=nd.array(data_int8_i), weight=nd.array(weight_int8_i),
                                                      bias=None, kernel=kernel, pad=pad, stride=stride,
                                                      num_filter=1, no_bias=True, num_group=1)

                    # dequantize
                    # scale_rate = sf_in * scaling_factor_i  # / sf_out
                    if (self.writefile):
                        scale_rate_fp32[i] = (sf_in * scaling_factor_i) / sf_out
                        weight_qt8[i] = weight_int8_i
                    conv_i = top_blob_int32_i * sf_in * scaling_factor_i
                    conv[0][i] = conv_i[0][0] + self.biases[id][i]

                # depthwise
                if (self.writefile):
                    self.write_file_output(input, 'input_' + layer_name + '.txt')
                    self.write_file_output(data_int8, 'input_qt8_' + layer_name + '.txt')
                    # self.write_file(self.weights[id], 'weight_' + layer_name + '.txt')
                    # self.write_file(self.biases[id], 'bias_' + layer_name + '.txt', bias=True)
                    # self.write_file(weight_qt8.asnumpy(), 'weight_qt8_' + layer_name + '.txt')
                    # self.write_file(bias_qt8.asnumpy(), 'bias_qt8_' + layer_name + '.txt', bias=True)
                    # self.write_file(scale_rate_fp32.asnumpy(), 'scale_rate_fp32_' + layer_name + '.txt', bias=True)

        else:
            if (no_bias==True or self.no_bias==True):
                b=None
            else:
                b=nd.array(self.biases[id])

            begin = time.time()
            conv = nd.Convolution(data=nd.array(input), weight=nd.array(self.weights[id]),bias=b, kernel=kernel, pad=pad, stride=stride,
                                  num_filter=filter, no_bias=no_bias, num_group=group)
            process = time.time() - begin

            self.timing[id] += process
        if (self.relu[id] == 0):
            final_result=conv.asnumpy()
        else:
            final_result=nd.relu(conv).asnumpy()
        if (self.writefile):
            self.write_file_output(final_result, 'output_' + layer_name + '.txt')
        return final_result

    def get_mbox_conf(self, mbox_1, mbox_2, mbox_3, mbox_4, mbox_5, mbox_6, transpose=True):
        if(transpose==True):
            mbox_1 = mbox_1.transpose(0, 2, 3, 1)
            mbox_2 = mbox_2.transpose(0, 2, 3, 1)
            mbox_3 = mbox_3.transpose(0, 2, 3, 1)
            mbox_4 = mbox_4.transpose(0, 2, 3, 1)
            mbox_5 = mbox_5.transpose(0, 2, 3, 1)
            mbox_6 = mbox_6.transpose(0, 2, 3, 1)
        flatten1 = mbox_1.reshape(1, mbox_1.shape[1] * mbox_1.shape[2] * mbox_1.shape[3] * mbox_1.shape[0])
        flatten2 = mbox_2.reshape(1, mbox_2.shape[1] * mbox_2.shape[2] * mbox_2.shape[3] * mbox_2.shape[0])
        flatten3 = mbox_3.reshape(1, mbox_3.shape[1] * mbox_3.shape[2] * mbox_3.shape[3] * mbox_3.shape[0])
        flatten4 = mbox_4.reshape(1, mbox_4.shape[1] * mbox_4.shape[2] * mbox_4.shape[3] * mbox_4.shape[0])
        flatten5 = mbox_5.reshape(1, mbox_5.shape[1] * mbox_5.shape[2] * mbox_5.shape[3] * mbox_5.shape[0])
        flatten6 = mbox_6.reshape(1, mbox_6.shape[1] * mbox_6.shape[2] * mbox_6.shape[3] * mbox_6.shape[0])
        concat = np.concatenate((flatten1, flatten2, flatten3, flatten4, flatten5, flatten6), axis=1)
        reshape = concat.reshape(1, 1917, 21)
        transpose = reshape.transpose(0, 2, 1)
        softmax = nd.SoftmaxActivation(data=nd.array(transpose), mode='channel')
        return softmax

    def get_mbox_loc(self,mbox_1, mbox_2, mbox_3, mbox_4, mbox_5, mbox_6, transpose= True):
        if(transpose==True):
            mbox_1 = mbox_1.transpose(0, 2, 3, 1)
            mbox_2 = mbox_2.transpose(0, 2, 3, 1)
            mbox_3 = mbox_3.transpose(0, 2, 3, 1)
            mbox_4 = mbox_4.transpose(0, 2, 3, 1)
            mbox_5 = mbox_5.transpose(0, 2, 3, 1)
            mbox_6 = mbox_6.transpose(0, 2, 3, 1)
        flatten1 = mbox_1.reshape(1, mbox_1.shape[1] * mbox_1.shape[2] * mbox_1.shape[3] * mbox_1.shape[0])
        flatten2 = mbox_2.reshape(1, mbox_2.shape[1] * mbox_2.shape[2] * mbox_2.shape[3] * mbox_2.shape[0])
        flatten3 = mbox_3.reshape(1, mbox_3.shape[1] * mbox_3.shape[2] * mbox_3.shape[3] * mbox_3.shape[0])
        flatten4 = mbox_4.reshape(1, mbox_4.shape[1] * mbox_4.shape[2] * mbox_4.shape[3] * mbox_4.shape[0])
        flatten5 = mbox_5.reshape(1, mbox_5.shape[1] * mbox_5.shape[2] * mbox_5.shape[3] * mbox_5.shape[0])
        flatten6 = mbox_6.reshape(1, mbox_6.shape[1] * mbox_6.shape[2] * mbox_6.shape[3] * mbox_6.shape[0])
        concat = np.concatenate((flatten1, flatten2, flatten3, flatten4, flatten5, flatten6), axis=1)
        return concat

    def get_mbox_prior(self,priorBox_1, priorBox_2, priorBox_3, priorBox_4, priorBox_5, priorBox_6):
        if('caffe' in self.model_version):
            steps=[-1.,-1.,-1.,-1.,-1.,-1.]
        if('mxnet' in self.model_version):
            steps = [16., 32., 64., 128., 256.,512.]

        output_priorBox_1 = self.get_prior_output(priorBox_1, sizes=(0.2,), ratios=(1.0, 2.0, 0.5), steps=(steps[0]/300, steps[0]/300))
        output_priorBox_2 = self.get_prior_output(priorBox_2, sizes=(0.35, 0.41833), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(steps[1]/300, steps[1]/300))
        output_priorBox_3 = self.get_prior_output(priorBox_3, sizes=(0.5, 0.570088), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(steps[2]/300, steps[2]/300))
        output_priorBox_4 = self.get_prior_output(priorBox_4, sizes=(0.65, 0.72111), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(steps[3]/300, steps[3]/300))
        output_priorBox_5 = self.get_prior_output(priorBox_5, sizes=(0.8, 0.87178), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(steps[4]/300, steps[4]/300))
        output_priorBox_6 = self.get_prior_output(priorBox_6, sizes=(0.95, 0.974679), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(steps[5]/300, steps[5]/300))

        flatten1 = output_priorBox_1.reshape(1, output_priorBox_1.shape[1] * output_priorBox_1.shape[2])
        flatten2 = output_priorBox_2.reshape(1, output_priorBox_2.shape[1] * output_priorBox_2.shape[2])
        flatten3 = output_priorBox_3.reshape(1, output_priorBox_3.shape[1] * output_priorBox_3.shape[2])
        flatten4 = output_priorBox_4.reshape(1, output_priorBox_4.shape[1] * output_priorBox_4.shape[2])
        flatten5 = output_priorBox_5.reshape(1, output_priorBox_5.shape[1] * output_priorBox_5.shape[2])
        flatten6 = output_priorBox_6.reshape(1, output_priorBox_6.shape[1] * output_priorBox_6.shape[2])
        concat = np.concatenate((flatten1, flatten2, flatten3, flatten4, flatten5, flatten6), axis=1)
        multibox_prior = concat.reshape(1, 1917, 4)

        return multibox_prior

    def get_prior_output(self,input_data, sizes=(1,), ratios=(1,), steps=(-1, -1), offsets=(0.5, 0.5), clip=False):
        dshape = input_data.shape
        dtype = 'float32'

        in_height = dshape[2]
        in_width = dshape[3]
        num_sizes = len(sizes)
        num_ratios = len(ratios)
        size_ratio_concat = sizes + ratios
        steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
        steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
        offset_h = offsets[0]
        offset_w = offsets[1]

        oshape = (1, in_height * in_width * (num_sizes + num_ratios - 1), 4)
        np_out = np.zeros(oshape).astype(dtype)

        for i in range(in_height):
            center_h = (i + offset_h) * steps_h
            for j in range(in_width):
                center_w = (j + offset_w) * steps_w
                for k in range(num_sizes + num_ratios - 1):
                    w = size_ratio_concat[k] * in_height / in_width / 2.0 if k < num_sizes else \
                        size_ratio_concat[0] * in_height / in_width * math.sqrt(size_ratio_concat[k + 1]) / 2.0
                    h = size_ratio_concat[k] / 2.0 if k < num_sizes else \
                        size_ratio_concat[0] / math.sqrt(size_ratio_concat[k + 1]) / 2.0
                    count = i * in_width * (num_sizes + num_ratios - 1) + j * (num_sizes + num_ratios - 1) + k
                    np_out[0][count][0] = center_w - w
                    np_out[0][count][1] = center_h - h
                    np_out[0][count][2] = center_w + w
                    np_out[0][count][3] = center_h + h
        if clip:
            np_out = np.clip(np_out, 0, 1)
        return np_out

    def get_detection_out_tvm(self,np_cls_prob, np_loc_preds, np_anchors, batch_size, num_anchors, num_classes):
        target_cpu = 'llvm'
        ctx = tvm.cpu()
        num_anchors = 1917
        cls_prob = tvm.placeholder((1, 21, num_anchors), name="cls_prob")
        loc_preds = tvm.placeholder((1, num_anchors*4), name="loc_preds")
        anchors = tvm.placeholder((1, num_anchors, 4), name="anchors")

        tvm_cls_prob = tvm.nd.array(np_cls_prob.asnumpy().astype(cls_prob.dtype), ctx)
        tvm_loc_preds = tvm.nd.array(np_loc_preds.astype(loc_preds.dtype), ctx)
        tvm_anchors = tvm.nd.array(np_anchors.astype(anchors.dtype), ctx)

        import topi
        with tvm.target.create(target_cpu):
            out = topi.vision.ssd.multibox_detection(cls_prob, loc_preds, anchors, clip=False, threshold=0.01,
                                                     nms_threshold=0.45,
                                                     force_suppress=False, variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
            s = topi.generic.schedule_multibox_detection(out)

        tvm_out = tvm.nd.array(np.zeros((1, num_anchors, 6)).astype(out.dtype), ctx)
        f = tvm.build(s, [cls_prob, loc_preds, anchors, out], 'llvm')
        f(tvm_cls_prob, tvm_loc_preds, tvm_anchors, tvm_out)
        return tvm_out

    def detect(self, image_data,img_name, no_bias=False, quantize=False):
        self.input_data=[]
        self.output_data=[]
        for i in range(self.total_layer):
            #print('id: '+str(i)+", layer: "+self.layer_names[i])
            if(i==0):
                self.input_data.append(image_data)
            else:
                self.input_data.append(self.output_data[int(self.input_layer[i])])
            self.output_data.append(self.convolution(self.input_data[i], i, no_bias=no_bias, quantize=quantize))

        print('mbox_conf')
        mbox_conf = self.get_mbox_conf(self.output_data[36], self.output_data[38], self.output_data[40], self.output_data[42],
                                 self.output_data[44], self.output_data[46])
        mbox_loc = self.get_mbox_loc(self.output_data[35], self.output_data[37], self.output_data[39], self.output_data[41],
                                  self.output_data[43], self.output_data[45])
        mbox_prior = self.get_mbox_prior(self.output_data[22], self.output_data[26], self.output_data[28], self.output_data[30],
                                        self.output_data[32], self.output_data[34])

        # mbox_conf=dict()
        # mbox_conf['conv11'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv11_mbox_conf/'+img_name+'.txt').reshape(1,63,19,19)
        # mbox_conf['conv13'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv13_mbox_conf/'+img_name+'.txt').reshape(1,126,10,10)
        # mbox_conf['conv14_2'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv14_2_mbox_conf/'+img_name+'.txt').reshape(1,126,5,5)
        # mbox_conf['conv15_2'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv15_2_mbox_conf/'+img_name+'.txt').reshape(1,126,3,3)
        # mbox_conf['conv16_2'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv16_2_mbox_conf/'+img_name+'.txt').reshape(1,126,2,2)
        # mbox_conf['conv17_2'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv17_2_mbox_conf/'+img_name+'.txt').reshape(1,126,1,1)
        #
        # mbox_loc=dict()
        # mbox_loc['conv11'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv11_mbox_loc/'+img_name+'.txt').reshape(1,12,19,19)
        # mbox_loc['conv13'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv13_mbox_loc/'+img_name+'.txt').reshape(1,24,10,10)
        # mbox_loc['conv14_2'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv14_2_mbox_loc/'+img_name+'.txt').reshape(1,24,5,5)
        # mbox_loc['conv15_2'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv15_2_mbox_loc/'+img_name+'.txt').reshape(1,24,3,3)
        # mbox_loc['conv16_2'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv16_2_mbox_loc/'+img_name+'.txt').reshape(1,24,2,2)
        # mbox_loc['conv17_2'] = self.read_data('/media/prdcv/Data/CuongND/output_toan_chaidnn_2203/conv17_2_mbox_loc/'+img_name+'.txt').reshape(1,24,1,1)

        #
        # mbox_conf = self.get_mbox_conf(mbox_conf['conv11'],mbox_conf['conv13'],mbox_conf['conv14_2'],mbox_conf['conv15_2'],mbox_conf['conv16_2'],mbox_conf['conv17_2'])
        # mbox_loc = self.get_mbox_loc(mbox_loc['conv11'],mbox_loc['conv13'],mbox_loc['conv14_2'],mbox_loc['conv15_2'],mbox_loc['conv16_2'],mbox_loc['conv17_2'])
        # mbox_prior=self.read_data('/home/prdcv/PycharmProjects/gvh205/from_Slack/post_processing/000001_mbox_priorbox.bin')[0:7668].reshape(1,1917,4)

        #t0=mbox_conf.asnumpy()
        #t1 = mbox_loc.asnumpy()
        #t2 = mbox_prior.asnumpy()
        output = self.get_detection_out_tvm(mbox_conf, mbox_loc, mbox_prior, 1, 1917, 21)
        return output
