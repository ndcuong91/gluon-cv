from mxnet import nd
from mxnet.gluon import nn

# layer = nn.Dense(2)
# layer.initialize()
#
# x = nd.random.uniform(-1,1,(3,4))
# output=layer(x)
# w=layer.weight.data()
#
#
# net = nn.Sequential()
# # Add a sequence of layers.
# net.add(# Similar to Dense, it is not necessary to specify the input channels
#         # by the argument `in_channels`, which will be  automatically inferred
#         # in the first forward pass. Also, we apply a relu activation on the
#         # output. In addition, we can use a tuple to specify a  non-square
#         # kernel size, such as `kernel_size=(2,4)`
#         nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
#         # One can also use a tuple to specify non-symmetric pool and stride sizes
#         nn.MaxPool2D(pool_size=2, strides=2),
#         nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         # The dense layer will automatically reshape the 4-D output of last
#         # max pooling layer into the 2-D shape: (x.shape[0], x.size/x.shape[0])
#         nn.Dense(120, activation="relu"),
#         nn.Dense(84, activation="relu"),
#         nn.Dense(10))
#
# net.initialize()
# # Input shape is (batch_size, color_channels, height, width)
# x = nd.random.uniform(shape=(4,1,28,28))
# y = net(x)

#
# class MixMLP(nn.Block):
#     def __init__(self, **kwargs):
#         # Run `nn.Block`'s init method
#         super(MixMLP, self).__init__(**kwargs)
#         self.blk = nn.Sequential()
#         self.blk.add(nn.Dense(3, activation='relu'),
#                      nn.Dense(4, activation='relu'))
#         self.dense = nn.Dense(5)
#     def forward(self, x):
#         y = nd.relu(self.blk(x))
#         print(y)
#         return self.dense(y)
#
# net = MixMLP()
# net.initialize()
# x = nd.random.uniform(shape=(2,2))
# net(x)
from mxnet import nd
from mxnet import autograd

x = nd.array([[1, 2], [3, 4]])
x.attach_grad()

with autograd.record():
    y = 2 * x * x
y.backward()
out=x.grad


from mxnet import nd
from mxnet import autograd

def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()



def train_neural_network():
    from mxnet import nd, gluon, init, autograd
    from mxnet.gluon import nn
    from mxnet.gluon.data.vision import datasets, transforms
    from IPython import display
    import matplotlib.pyplot as plt
    import time
    import mxnet as mx

    mnist_train = datasets.FashionMNIST(train=True)
    X, y = mnist_train[0]
    ('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y)
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    X, y = mnist_train[0:10]
    # plot images
    display.set_matplotlib_formats('svg')
    _, figs = plt.subplots(1, X.shape[0], figsize=(15, 15))
    # for f, x, yi in zip(figs, X, y):
    #     # 3D->2D by removing the last channel dim
    #     f.imshow(x.reshape((28, 28)).asnumpy())
    #     ax = f.axes
    #     ax.set_title(text_labels[int(yi)])
    #     ax.title.set_fontsize(14)
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)])
    mnist_train = mnist_train.transform_first(transformer)

    batch_size = 256
    train_data = gluon.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

    mnist_valid = gluon.data.vision.FashionMNIST(train=False)
    valid_data = gluon.data.DataLoader(
        mnist_valid.transform_first(transformer),
        batch_size=batch_size, num_workers=4)

    net = nn.Sequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dense(120, activation="relu"),
            nn.Dense(84, activation="relu"),
            nn.Dense(10))
    #net.collect_params().initialize(force_reinit=True, ctx=mx.gpu())
    net.initialize(init=init.Xavier(),ctx=mx.gpu())

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    for epoch in range(10):
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time.time()
        for data, label in train_data:
            # forward + backward
            data_list = gluon.utils.split_and_load(data, ctx_list=[mx.gpu()])
            label_list = gluon.utils.split_and_load(label, ctx_list=[mx.gpu()])
            with autograd.record():
                #output = net(X)
                loss = [softmax_cross_entropy(net(X), y) for X, y in zip(data_list, label_list)]
            for l in loss:
                l.backward()
            # update parameters
            trainer.step(batch_size)
            # calculate training metrics
            train_loss += sum([l.sum().asscalar() for l in loss])
            #train_acc += acc(output, label)
        # calculate validation accuracy
        #for data, label in valid_data:
        #    valid_acc += acc(net(data), label)
        print("Epoch %d: loss %.3f, in %.1f sec" % (
            epoch, train_loss / len(train_data), time.time() - tic))

    # from gluoncv.utils import export_block
    # export_block('test', net)
    #symbol.save('%s-symbol.json' % prefix)
    net.save_parameters('net.params')

    print 'Finish'

def predict_neural_network():
    from mxnet import nd
    from mxnet import gluon
    from mxnet.gluon import nn
    from mxnet.gluon.data.vision import datasets, transforms
    from IPython import display
    import matplotlib.pyplot as plt

    net = nn.Sequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dense(120, activation="relu"),
            nn.Dense(84, activation="relu"),
            nn.Dense(10))
    net.load_parameters('net.params')
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)])
    mnist_valid = datasets.FashionMNIST(train=False)
    X, y = mnist_valid[:10]
    preds = []
    for x in X:
        x = transformer(x).expand_dims(axis=0)
        pred = net(x).argmax(axis=1)
        preds.append(pred.astype('int32').asscalar())

    _, figs = plt.subplots(1, 10, figsize=(15, 15))
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    display.set_matplotlib_formats('svg')
    for f, x, yi, pyi in zip(figs, X, y, preds):
        f.imshow(x.reshape((28, 28)).asnumpy())
        ax = f.axes
        ax.set_title(text_labels[yi] + '\n' + text_labels[pyi])
        ax.title.set_fontsize(14)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

train_neural_network()
#predict_neural_network()
print 'Finish'