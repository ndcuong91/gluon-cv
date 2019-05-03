from mxnet import gluon, init, nd
from mxnet.gluon import nn


def get_arm_network(version, num_class, ctx):
    if(version=='3.3'):
        return arm_network_v3_3(num_class, ctx)
    if(version=='3.4'):
        return arm_network_v3_4(num_class, ctx)
    if(version=='4.1'):
        return arm_network_v4_1(num_class, ctx)
    if(version=='4.2'):
        return arm_network_v4_2(num_class, ctx)
    if(version=='3.5'):
        return arm_network_v3_5(num_class, ctx)
    if(version=='3.5.1'):
        return arm_network_v3_5_1(num_class, ctx)
    if(version=='3.5.2'):
        return arm_network_v3_5_2(num_class, ctx)
    if(version=='3.5.3'):
        return arm_network_v3_5_3(num_class, ctx)
    if(version=='3.5.4'):
        return arm_network_v3_5_4(num_class, ctx)
    if(version=='3.6'):
        return arm_network_v3_6(num_class, ctx)
    if(version=='3.7'):
        return arm_network_v3_7(num_class, ctx)
    if(version=='3.8'):
        return arm_network_v3_8(num_class, ctx)
    if(version=='3.9'):
        return arm_network_v3_9(num_class, ctx)
    if(version=='3.10'):
        return arm_network_v3_10(num_class, ctx)

def arm_network_v3_3(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=8, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=16, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1),  # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=96, kernel_size=3, padding=1),  # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),

        nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(64, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_4(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=8, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=16, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1),  # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),

        nn.Conv2D(channels=96, kernel_size=3, padding=1),  # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_5(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=8, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=16, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        # nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_5_1(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=8, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        # nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_5_2(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=8, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        # nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_5_3(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=8, kernel_size=3, padding=1, use_bias=False), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1, use_bias=False), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1, use_bias=False), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=96, kernel_size=3, padding=1, use_bias=False), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1, use_bias=False), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        # nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_5_4(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=8, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=16, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=256, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=512, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        #nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_6(num_class, ctx): #global max pool
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=8, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=16, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.GlobalMaxPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_7(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=10, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=20, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=40, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=80, kernel_size=3, padding=1), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        # nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_8(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=16, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=256, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        # nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network

def arm_network_v3_9(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=6, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=12, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=24, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=48, kernel_size=3, padding=1), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=96, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        # nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network


def arm_network_v3_10(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=4, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=8, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=16, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1), # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        #nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=160, kernel_size=3, padding=1), # conv6
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network
def arm_network_v4_1(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=256, kernel_size=3, padding=1),  # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=256, kernel_size=3, padding=1),  # conv5
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),

        nn.Conv2D(channels=512, kernel_size=3, padding=1), # conv6
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(128, activation="relu"),
        nn.Dense(64, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network


def arm_network_v4_2(num_class, ctx):
    network = nn.HybridSequential()
    network.add(
        nn.Conv2D(channels=16, kernel_size=3, padding=1), # conv1
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=32, kernel_size=3, padding=1), # conv2
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=64, kernel_size=3, padding=1), # conv3
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Conv2D(channels=128, kernel_size=3, padding=1),  # conv4
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        # nn.Conv2D(channels=128, kernel_size=3, padding=1),  # conv5
        # nn.BatchNorm(axis=1, center=True, scale=True),
        # nn.Activation(activation='relu'),

        nn.Conv2D(channels=256, kernel_size=3, padding=1), # conv6
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Activation(activation='relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(64, activation="relu"),
        nn.Dense(32, activation="relu"),
        nn.Dense(num_class))
    network.initialize(init=init.Xavier(), ctx=ctx)
    return network