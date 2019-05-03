import caffe
import mxnet as mx

#CuongND. This tool use to parse params from Caffe to Mxnet's model. As gluon-cv code of 05/04/2019
#The parse params include batchnorm params (beta, gamma, running_mean, running_var, moving_mean, moving_var)

CAFFE_NET = '/home/atsg/PycharmProjects/gvh205/MobileNet-SSD-Caffe/deploy.prototxt'
CAFFE_MODEL = '/home/atsg/PycharmProjects/gvh205/MobileNet-SSD-Caffe/mobilenet_iter_73000.caffemodel' #trained model from Caffe with batchnorm layer

MXNET_MODEL = 'Begin_train_mobilenetSSD_same_as_Caffe/ssd_300_mobilenet1.0_voc_0000_0.0626.params' #only for reference
MXNET_SAVED='mobilenet_v1_300_rebuild_similar_parsed_fromCaffe.params'

layer_names=[
'conv0',
'conv1/dw',
'conv1',
'conv2/dw',
'conv2',
'conv3/dw',
'conv3',
'conv4/dw',
'conv4',
'conv5/dw',
'conv5',
'conv6/dw',
'conv6',
'conv7/dw',
'conv7',
'conv8/dw',
'conv8',
'conv9/dw',
'conv9',
'conv10/dw',
'conv10',
'conv11/dw',
'conv11',
'conv12/dw,'
'conv12',
'conv13/dw',
'conv13',
'conv14_1',
'conv14_2',
'conv15_1',
'conv15_2',
'conv16_1',
'conv16_2',
'conv17_1',
'conv17_2',
'conv11_mbox_loc',
'conv11_mbox_conf',
'conv13_mbox_loc',
'conv13_mbox_conf',
'conv14_2_mbox_loc',
'conv14_2_mbox_conf',
'conv15_2_mbox_loc',
'conv15_2_mbox_conf',
'conv16_2_mbox_loc',
'conv16_2_mbox_conf',
'conv17_2_mbox_loc',
'conv17_2_mbox_conf'
]

caffe_net = caffe.Net(CAFFE_NET, CAFFE_MODEL, caffe.TEST)
mbox=dict()
mbox['11']=0
mbox['13']=1
mbox['14_2']=2
mbox['15_2']=3
mbox['16_2']=4
mbox['17_2']=5
def get_params(cnet, mx_params):
    arg_params_map = mx_params.copy()
    expand=False
    for caffe_layer_name, blob_vec in cnet.params.iteritems():
        print
        print caffe_layer_name
        conv='conv' in caffe_layer_name
        dw ='dw' in caffe_layer_name
        bn='bn' in caffe_layer_name
        scale='scale' in caffe_layer_name
        mbox_='mbox' in caffe_layer_name
        conf='conf' in caffe_layer_name
        layer_id = caffe_layer_name.replace('/dw', '').replace('/bn', '').replace('conv', '').replace('_mbox', '').replace('_loc', '').replace('_conf', '')
        if (conv==True and bn == False and scale==False): #conv or conv/dw
            if(mbox_==True):
                if(conf==True):
                    w_name='class_predictors.'+str(mbox[layer_id])+'.predictor.weight'
                    b_name='class_predictors.'+str(mbox[layer_id])+'.predictor.bias'
                else: #loc
                    w_name='box_predictors.'+str(mbox[layer_id])+'.predictor.weight'
                    b_name='box_predictors.'+str(mbox[layer_id])+'.predictor.bias'

                print 'Parse weight and bias'
                arg_params_map[w_name] = mx.nd.array(blob_vec[0].data)
                arg_params_map[b_name] = mx.nd.array(blob_vec[1].data)

            else:
                if (layer_id == '14_1'):
                    expand = True
                if (expand == False):
                    if (dw == True):
                        layer_name = 'conv' + str(2 * int(layer_id) - 1)
                    else:
                        layer_name = 'conv' + str(2 * int(layer_id))
                    arg_name = 'features.mobilenet0_' + layer_name + '_weight'
                else:
                    arg_name = 'features.expand_conv' + layer_id + '_weight'

                print 'Parse weight'
                arg_params_map[arg_name] = mx.nd.array(blob_vec[0].data)


        elif (bn==True):
            if(expand==False):
                if (dw == True):
                    layer_name = str(2 * int(layer_id) - 1)
                else:
                    layer_name = str(2 * int(layer_id))
                aux_name = 'features.mobilenet0_batchnorm' + layer_name
            else:
                aux_name = 'features.expand_bn' + layer_id

            if(expand==False):
                print 'Parse _running_mean and _running_var'
                arg_params_map[aux_name + '_running_mean'] = mx.nd.array(blob_vec[0].data)
                arg_params_map[aux_name + '_running_var'] = mx.nd.array(blob_vec[1].data)
            else:
                print 'Parse _moving_mean and _moving_var'
                arg_params_map[aux_name + '_moving_mean'] = mx.nd.array(blob_vec[0].data)
                arg_params_map[aux_name + '_moving_var'] = mx.nd.array(blob_vec[1].data)

            print 'Parse gamma and beta'
            arg_params_map[aux_name + '_gamma'] = mx.nd.array(cnet.params[caffe_layer_name.replace('bn', 'scale')][0].data)
            arg_params_map[aux_name + '_beta'] = mx.nd.array(cnet.params[caffe_layer_name.replace('bn', 'scale')][1].data)

        elif (scale==True): #scale
            assert caffe_layer_name.replace('scale', 'bn') in cnet.params
        else:
            raise ValueError

    return arg_params_map

def rename_anchors(params):

    params['anchor_generators.0.anchors']=params['ssdanchorgenerator0_anchor_0']
    params['anchor_generators.1.anchors']=params['ssdanchorgenerator1_anchor_1']
    params['anchor_generators.2.anchors']=params['ssdanchorgenerator2_anchor_2']
    params['anchor_generators.3.anchors']=params['ssdanchorgenerator3_anchor_3']
    params['anchor_generators.4.anchors']=params['ssdanchorgenerator4_anchor_4']
    params['anchor_generators.5.anchors']=params['ssdanchorgenerator5_anchor_5']

    params.pop('ssdanchorgenerator0_anchor_0', None)
    params.pop('ssdanchorgenerator1_anchor_1', None)
    params.pop('ssdanchorgenerator2_anchor_2', None)
    params.pop('ssdanchorgenerator3_anchor_3', None)
    params.pop('ssdanchorgenerator4_anchor_4', None)
    params.pop('ssdanchorgenerator5_anchor_5', None)


loaded = mx.ndarray.load(MXNET_MODEL)
arg_params_map = get_params(caffe_net, loaded)
rename_anchors(arg_params_map)

save_dict = {('%s' % k) : v for k, v in arg_params_map.items()}
mx.nd.save(MXNET_SAVED, save_dict)

print ('Save to '+MXNET_SAVED+' !')
