import numpy as np
import os

input_dir='/home/atsg/PycharmProjects/gvh205/arm_project/arm_data/arm_v4.5.3'
model='arm_v4.5.3'
saved_folder=os.path.join('arm_data',model)

layers=['conv1','conv2','conv3','conv4','conv5','conv6']

def read_data(filename, len_to_read=-1):  # read .bin or .txt data file
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


def write_file(data, filename, folder=saved_folder, txt=True, inter=True, separate='\n'):
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
            if(inter):
                text += str(int(data[i])) + separate
            else:
                text += str(data[i]) + separate

        with open(os.path.join(folder, filename), "w") as text_file:
            text_file.write(text)
    else:
        data.tofile(os.path.join(folder, filename))

def save_data():

    scalein_path=os.path.join(input_dir,'scaleLayerIn.txt')
    data_prev_path=os.path.join(input_dir,'data_scale_prev.txt')
    data_scale_prev=read_data(data_prev_path)
    scalein=data_scale_prev/127
    write_file(scalein,scalein_path,inter=False)

    scaleout_path=os.path.join(input_dir,'scaleLayerOut.txt')
    data_path=os.path.join(input_dir,'data_scale.txt')
    data_scale=read_data(data_path)
    scaleout=data_scale/127
    write_file(scaleout,scaleout_path,inter=False)

    #gen weight
    for layer in layers:
        print layer
        weight_qt_path = os.path.join(input_dir, layer + '_weight_qt.txt')
        th_params_path = os.path.join(input_dir, layer + '_weight_scale.txt')
        weight_path = os.path.join(input_dir, layer + '_weight.txt')
        weight = read_data(weight_path)
        th_params= read_data(th_params_path)
        weight_qt=[]
        total_weight=len(weight)
        total_th_param=len(th_params)
        weight_per_channel=total_weight/total_th_param
        for i in range(total_weight):
            #print i
            channel=(int)(i/weight_per_channel)
            print channel
            temp=(127*weight[i])/(th_params[channel])
            weight_qt.append(temp.round())
        write_file(np.asarray(weight_qt), weight_qt_path, separate=',')


    #gen bias
    count=0
    for layer in layers:
        print layer
        bias_qt_path = os.path.join(input_dir, layer + '_bias_qt.txt')
        bias_path = os.path.join(input_dir, layer + '_bias.txt')
        bias = read_data(bias_path)
        bias_qt=[]
        total_bias=len(bias)
        for i in range(total_bias):
            #print i
            temp=(127*bias[i])/(data_scale[count])
            bias_qt.append(temp.round())
        count+=1
        write_file(np.asarray(bias_qt), bias_qt_path, separate=',')

    #gen scale rate
    count=0
    for layer in layers:
        print layer
        th_params_path = os.path.join(input_dir, layer + '_weight_scale.txt')
        th_params= read_data(th_params_path)
        scaleRate_path = os.path.join(input_dir, layer + '_scaleRate.txt')
        scaleRate_fp32=[]
        total_th_param=len(th_params)
        for i in range(total_th_param):
            #print i
            temp=(scalein[count]*th_params[i])/scaleout[count]
            scaleRate_fp32.append(temp)
        write_file(np.asarray(scaleRate_fp32), scaleRate_path,inter=False, separate=',')

    print 'OK'


if __name__ == "__main__":
    save_data()

    print 'Finish'