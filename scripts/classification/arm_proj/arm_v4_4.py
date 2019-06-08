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
        self.weight_folder=weight_folder

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
        print('Arm_v4_4.py. Finish initialize parameters.')

    def write_file(data, filename, folder='data_arm_v4.4', txt=True, inter=True, separate='\n'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        length = len(data.shape)
        total_value = 1
        for i in range(length):
            total_value *= data.shape[i]

        data = data.reshape(total_value)
        if (txt == True):
            text = ''
            for i in range(total_value):
                if (inter):
                    text += str(int(data[i])) + separate
                else:
                    text += str(data[i]) + separate

            with open(os.path.join(folder, filename), "w") as text_file:
                text_file.write(text)
        else:
            data.tofile(os.path.join(folder, filename))

    def write_file_output(self, data,filename, folder='data_arm_v4.4', txt=True): #txt or bin file
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

    def get_weight(self, weight_folder):
        for i in range(self.total_layer):
            if ('pool' in self.layer_names[i]):
                self.weights.append(-1)
                continue

            layer_name=self.layer_names[i].replace('/','_')
            print('get weight in ' +layer_name)
            weight = np.fromfile(os.path.join(weight_folder, layer_name + '_weight.bin'), dtype='float32')
            weight=np.reshape(weight,self.weight_shapes[i])
            #weight=weight.transpose((0, 2, 3, 1))
            self.weights.append(np.reshape(weight,self.weight_shapes[i]))

    def get_bias(self, bias_folder):
        for i in range(self.total_layer):
            if ('pool' in self.layer_names[i]):
                self.biases.append(-1)
                continue
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

        if (quantize == True):
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
                                                  num_filter=1, no_bias=True, num_group=1)

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
                self.write_file(weight_qt8.asnumpy(), layer_name + '_weight_qt.txt')
                self.write_file(bias_qt8.asnumpy(), layer_name + 'bias_qt.txt', bias=True)
                #self.write_file(scale_rate_fp32.asnumpy(), 'sf_out_fp32_' + layer_name + '.txt', bias=True)
                self.write_file(scale_rate_fp32.asnumpy(), layer_name + '_scaleRate.txt', bias=True)

        else:
            if (no_bias==True or self.no_bias==True):
                b=None
            else:
                b=nd.array(self.biases[id])

            conv = nd.Convolution(data=nd.array(input), weight=nd.array(self.weights[id]),bias=b, kernel=kernel, pad=pad, stride=stride,
                                  num_filter=filter, no_bias=no_bias, num_group=1)
        if (self.relu[id] == 0):
            final_result=conv.asnumpy()
        else:
            final_result=nd.relu(conv).asnumpy()
        if (self.writefile):
            self.write_file_output(final_result, 'output_' + layer_name + '.txt')
        return final_result

    def pool(self, input, id, pool_type='max', global_pool=False):
        if(global_pool==False):
            pad=(0,0)
            if(self.layer_names[id]=='maxpool3' or self.layer_names[id]=='maxpool4'):
                pad=(1,1)
            output= nd.Pooling(data=nd.array(input), kernel=(2,2), stride=(2,2), pad=pad, pool_type='max', global_pool=global_pool)
        else:
            output= nd.Pooling(data=nd.array(input), pool_type=pool_type, global_pool=global_pool)
        final_result = output.asnumpy()
        return final_result

    def dense(self,layer_name, input, weight, bias, num_hidden):
        output=nd.FullyConnected(data=nd.array(input),weight=nd.array(weight), bias=nd.array(bias), num_hidden=num_hidden)
        final_result = output.asnumpy()
        if (self.writefile):
            self.write_file(weight, layer_name + '_weight.txt')
            self.write_file(bias, layer_name + '_bias.txt')
        return final_result


    def classify(self, image_data,img_name, no_bias=False, quantize=False):
        self.input_data=[]
        self.output_data=[]
        for i in range(self.total_layer):
            #print('id: '+str(i)+", layer: "+self.layer_names[i])
            if(i==0):
                self.input_data.append(image_data)
            else:
                self.input_data.append(self.output_data[int(self.input_layer[i])])

            if ('conv' in self.layer_names[i]): #conv
                self.output_data.append(self.convolution(self.input_data[i], i, no_bias=no_bias, quantize=quantize))
            if ('maxpool' in self.layer_names[i]): #max pool
                self.output_data.append(self.pool(self.input_data[i], i, pool_type='max'))
            if ('global_avg_pool' in self.layer_names[i]): #global average pool
                self.output_data.append(self.pool(self.input_data[i], i, pool_type='avg', global_pool=True))

        weight_dense1 = np.fromfile(os.path.join(self.weight_folder, 'dense1_weight.bin'), dtype='float32').reshape(128,256)
        bias_dense1 = np.fromfile(os.path.join(self.weight_folder, 'dense1_bias.bin'), dtype='float32')
        output_dense1=self.dense('dense1', self.output_data[11],weight_dense1,bias_dense1,num_hidden=128)

        weight_dense2 = np.fromfile(os.path.join(self.weight_folder, 'dense2_weight.bin'), dtype='float32').reshape(64,128)
        bias_dense2 = np.fromfile(os.path.join(self.weight_folder, 'dense2_bias.bin'), dtype='float32')
        output_dense2 = self.dense('dense2',output_dense1, weight_dense2, bias_dense2, num_hidden=64)

        weight_dense3 = np.fromfile(os.path.join(self.weight_folder, 'dense3_weight.bin'), dtype='float32').reshape(2,64)
        bias_dense3 = np.fromfile(os.path.join(self.weight_folder, 'dense3_bias.bin'), dtype='float32')
        output_dense3 = self.dense('dense3',output_dense2, weight_dense3, bias_dense3, num_hidden=2)

        #print('Softmax')
        softmax = nd.SoftmaxActivation(data=nd.array(output_dense3))
        return softmax
