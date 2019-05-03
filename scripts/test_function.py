
import mxnet as mx
import gluoncv as gcv
from gluoncv.utils import export_block

pretrained='train_mobilenetSSD_same_as_Caffe_parse_params_7337/ssd_300_mobilenet1.0_voc_same_as_caffe_7337.params'


def test_nms():
    x = [[0, 0.5, 0.1, 0.1, 0.2, 0.2], [1, 0.4, 0.1, 0.1, 0.2, 0.2],
         [0, 0.3, 0.1, 0.1, 0.14, 0.14], [2, 0.6, 0.5, 0.5, 0.7, 0.8]]
    res = mx.contrib.box_nms(x, overlap_thresh=0.1, coord_start=2, score_index=1, id_index=0, force_suppress=True,
                             in_format='corner', out_typ='corner')

def export_network(name):
    net = gcv.model_zoo.get_model('ssd_300_mobilenet1.0_voc', pretrained=False, pretrained_base=False)
    net.load_parameters(pretrained)
    export_block(name, net)
    #shape = {'data': (100, 200)}
    #gluoncv.utils.viz.plot_network(net, shape={'data':(300,300)},save_prefix='abc')

    #gluoncv.nn.

def test_mxboard():
    from mxboard import SummaryWriter
    sw = SummaryWriter(logdir='./logs')
    import mxnet as mx
    for i in range(10):
        # create a normal distribution with fixed mean and decreasing std
        data = mx.nd.random.normal(loc=0, scale=10.0 / (i + 1), shape=(10, 3, 8, 8))
        sw.add_histogram(tag='norml_dist', values=data, bins=200, global_step=i)
    sw.close()



if __name__ == '__main__':
    #export_network('ssd_300_mobilenet1.0_voc')
    #deserialized_net = mx.gluon.nn.SymbolBlock.imports("ssd_300_mobilenet1.0_voc-symbol.json", ['data'], "ssd_300_mobilenet1.0_voc-0000.params", ctx=mx.cpu())
    test_mxboard()
print 'Finish'


