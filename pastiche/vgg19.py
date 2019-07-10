from collections import namedtuple
from typing import Iterable
import re

import h5py
import torch
import torch.nn as nn

# Another version of VGG19 is available from torchvision.

# The torchvision model matches the architecture of the original VGG, but the
# weights were retrained, which results in a different effect when using that
# model for style transfer (possibly due to not using the multi-scale training
# procedure used for the original model).

# The VGG19 class below loads arbitrary VGG19 weights. The original trained model
# was released as a Caffe model, but was subsequently released as a Keras model which
# is used for loading the weights below (it's an h5 file) using from_keras_h5.

# The Keras model file was released into the public domain from Kaggle:
#   https://www.kaggle.com/keras/vgg19/home
#   https://www.kaggle.com/keras/vgg19/downloads/
#           vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5/2

# It is also seemingly available at:
#   https://github.com/fchollet/deep-learning-models/releases/download/v0.1/
#           vgg19_weights_tf_dim_ordering_tf_kernels.h5
# That link is from:
#   https://github.com/keras-team/keras-applications/blob/master/
#           keras_applications/vgg19.py

# To generate a quantized model, load the original model, then call VGG19.save_quantized_bin.
#   > vgg19 = VGG19.from_keras_h5(keras_h5_path)
#   > vgg19.save_quantized_bin(quantized_bin_path)
# VGG19.from_quantized_bin loads a quantized model.
#   > vgg19_q = VGG19.from_quantized_bin(quantized_bin_path)


class VGG19(nn.Module):
    LAYER_NAMES = (
        'block1_conv1',
        'block1_relu1',
        'block1_conv2',
        'block1_relu2',
        'block1_pool',

        'block2_conv1',
        'block2_relu1',
        'block2_conv2',
        'block2_relu2',
        'block2_pool',

        'block3_conv1',
        'block3_relu1',
        'block3_conv2',
        'block3_relu2',
        'block3_conv3',
        'block3_relu3',
        'block3_conv4',
        'block3_relu4',
        'block3_pool',

        'block4_conv1',
        'block4_relu1',
        'block4_conv2',
        'block4_relu2',
        'block4_conv3',
        'block4_relu3',
        'block4_conv4',
        'block4_relu4',
        'block4_pool',

        'block5_conv1',
        'block5_relu1',
        'block5_conv2',
        'block5_relu2',
        'block5_conv3',
        'block5_relu3',
        'block5_conv4',
        'block5_relu4',
        'block5_pool',
    )

    Weights = namedtuple('Weights', [
        'block1_conv1_W', 'block1_conv1_b',
        'block1_conv2_W', 'block1_conv2_b',

        'block2_conv1_W', 'block2_conv1_b',
        'block2_conv2_W', 'block2_conv2_b',

        'block3_conv1_W', 'block3_conv1_b',
        'block3_conv2_W', 'block3_conv2_b',
        'block3_conv3_W', 'block3_conv3_b',
        'block3_conv4_W', 'block3_conv4_b',

        'block4_conv1_W', 'block4_conv1_b',
        'block4_conv2_W', 'block4_conv2_b',
        'block4_conv3_W', 'block4_conv3_b',
        'block4_conv4_W', 'block4_conv4_b',

        'block5_conv1_W', 'block5_conv1_b',
        'block5_conv2_W', 'block5_conv2_b',
        'block5_conv3_W', 'block5_conv3_b',
        'block5_conv4_W', 'block5_conv4_b',
    ])

    def __init__(self, weights):
        super(VGG19, self).__init__()

        # Layer specifications

        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Weight instantiation

        self.block1_conv1.weight.data = torch.from_numpy(weights.block1_conv1_W)
        self.block1_conv1.bias.data = torch.from_numpy(weights.block1_conv1_b)
        self.block1_conv2.weight.data = torch.from_numpy(weights.block1_conv2_W)
        self.block1_conv2.bias.data = torch.from_numpy(weights.block1_conv2_b)

        self.block2_conv1.weight.data = torch.from_numpy(weights.block2_conv1_W)
        self.block2_conv1.bias.data = torch.from_numpy(weights.block2_conv1_b)
        self.block2_conv2.weight.data = torch.from_numpy(weights.block2_conv2_W)
        self.block2_conv2.bias.data = torch.from_numpy(weights.block2_conv2_b)

        self.block3_conv1.weight.data = torch.from_numpy(weights.block3_conv1_W)
        self.block3_conv1.bias.data = torch.from_numpy(weights.block3_conv1_b)
        self.block3_conv2.weight.data = torch.from_numpy(weights.block3_conv2_W)
        self.block3_conv2.bias.data = torch.from_numpy(weights.block3_conv2_b)
        self.block3_conv3.weight.data = torch.from_numpy(weights.block3_conv3_W)
        self.block3_conv3.bias.data = torch.from_numpy(weights.block3_conv3_b)
        self.block3_conv4.weight.data = torch.from_numpy(weights.block3_conv4_W)
        self.block3_conv4.bias.data = torch.from_numpy(weights.block3_conv4_b)

        self.block4_conv1.weight.data = torch.from_numpy(weights.block4_conv1_W)
        self.block4_conv1.bias.data = torch.from_numpy(weights.block4_conv1_b)
        self.block4_conv2.weight.data = torch.from_numpy(weights.block4_conv2_W)
        self.block4_conv2.bias.data = torch.from_numpy(weights.block4_conv2_b)
        self.block4_conv3.weight.data = torch.from_numpy(weights.block4_conv3_W)
        self.block4_conv3.bias.data = torch.from_numpy(weights.block4_conv3_b)
        self.block4_conv4.weight.data = torch.from_numpy(weights.block4_conv4_W)
        self.block4_conv4.bias.data = torch.from_numpy(weights.block4_conv4_b)

        self.block5_conv1.weight.data = torch.from_numpy(weights.block5_conv1_W)
        self.block5_conv1.bias.data = torch.from_numpy(weights.block5_conv1_b)
        self.block5_conv2.weight.data = torch.from_numpy(weights.block5_conv2_W)
        self.block5_conv2.bias.data = torch.from_numpy(weights.block5_conv2_b)
        self.block5_conv3.weight.data = torch.from_numpy(weights.block5_conv3_W)
        self.block5_conv3.bias.data = torch.from_numpy(weights.block5_conv3_b)
        self.block5_conv4.weight.data = torch.from_numpy(weights.block5_conv4_W)
        self.block5_conv4.bias.data = torch.from_numpy(weights.block5_conv4_b)

    def forward(self, input: torch.Tensor, output_layers: Iterable=LAYER_NAMES) -> dict:
        output = {}

        x = input

        x = output['block1_conv1'] = self.block1_conv1.forward(x)
        x = output['block1_relu1'] = self.relu.forward(x)
        x = output['block1_conv2'] = self.block1_conv2.forward(x)
        x = output['block1_relu2'] = self.relu.forward(x)
        x = output['block1_pool'] = self.pool.forward(x)

        x = output['block2_conv1'] = self.block2_conv1.forward(x)
        x = output['block2_relu1'] = self.relu.forward(x)
        x = output['block2_conv2'] = self.block2_conv2.forward(x)
        x = output['block2_relu2'] = self.relu.forward(x)
        x = output['block2_pool'] = self.pool.forward(x)

        x = output['block3_conv1'] = self.block3_conv1.forward(x)
        x = output['block3_relu1'] = self.relu.forward(x)
        x = output['block3_conv2'] = self.block3_conv2.forward(x)
        x = output['block3_relu2'] = self.relu.forward(x)
        x = output['block3_conv3'] = self.block3_conv3.forward(x)
        x = output['block3_relu3'] = self.relu.forward(x)
        x = output['block3_conv4'] = self.block3_conv4.forward(x)
        x = output['block3_relu4'] = self.relu.forward(x)
        x = output['block3_pool'] = self.pool.forward(x)

        x = output['block4_conv1'] = self.block4_conv1.forward(x)
        x = output['block4_relu1'] = self.relu.forward(x)
        x = output['block4_conv2'] = self.block4_conv2.forward(x)
        x = output['block4_relu2'] = self.relu.forward(x)
        x = output['block4_conv3'] = self.block4_conv3.forward(x)
        x = output['block4_relu3'] = self.relu.forward(x)
        x = output['block4_conv4'] = self.block4_conv4.forward(x)
        x = output['block4_relu4'] = self.relu.forward(x)
        x = output['block4_pool'] = self.pool.forward(x)

        x = output['block5_conv1'] = self.block5_conv1.forward(x)
        x = output['block5_relu1'] = self.relu.forward(x)
        x = output['block5_conv2'] = self.block5_conv2.forward(x)
        x = output['block5_relu2'] = self.relu.forward(x)
        x = output['block5_conv3'] = self.block5_conv3.forward(x)
        x = output['block5_relu3'] = self.relu.forward(x)
        x = output['block5_conv4'] = self.block5_conv4.forward(x)
        x = output['block5_relu4'] = self.relu.forward(x)
        x = output['block5_pool'] = self.pool.forward(x)

        output = {key: value for key, value in output.items() if key in output_layers}

        if not input.requires_grad:
            for key in output.keys():
                output[key] = output[key].detach()

        return output

    def save_quantized_bin(self, path):
        import kmeans1d  # Not required for general pastiche usage, just for generating quantized model.
        k = 2 ** 8
        q_state = {}  # quantized state
        layer_names = [layer_name for layer_name in VGG19.LAYER_NAMES if re.match('^block\d+_conv\d+$', layer_name)]
        for layer_name in layer_names:
            layer = getattr(self, layer_name)
            bias = layer.bias
            shape = layer.weight.shape
            weight = layer.weight.flatten()
            clusters, centroids = kmeans1d.cluster(weight, k)
            q_state[layer_name + '_W_q'] = torch.tensor(clusters, dtype=torch.uint8).reshape(shape)
            q_state[layer_name + '_W_table'] = torch.tensor(centroids, dtype=torch.float32)
            q_state[layer_name + '_b'] = bias.detach().to('cpu', copy=True)
        torch.save(q_state, path)

    @staticmethod
    def from_quantized_bin(path):
        q_state = torch.load(path, map_location='cpu')
        weights_dict = {}
        layer_names = [layer_name for layer_name in VGG19.LAYER_NAMES if re.match('^block\d+_conv\d+$', layer_name)]
        for layer_name in layer_names:
            W_q = q_state[layer_name + '_W_q']
            shape = W_q.shape
            W_table = q_state[layer_name + '_W_table']
            W = W_table[W_q.flatten().tolist()].reshape(shape)
            b = q_state[layer_name + '_b']
            weights_dict[layer_name + '_W'] = W.numpy()
            weights_dict[layer_name + '_b'] = b.numpy()
        weights = VGG19.Weights(**weights_dict)
        return VGG19(weights)

    @staticmethod
    def from_keras_h5(path):
        W_order = (3,2,0,1)
        with h5py.File(path, 'r') as f:
            weights = VGG19.Weights(
                block1_conv1_W=f['/block1_conv1/block1_conv1_W_1:0'][()].transpose(W_order),
                block1_conv1_b=f['/block1_conv1/block1_conv1_b_1:0'][()],
                block1_conv2_W=f['/block1_conv2/block1_conv2_W_1:0'][()].transpose(W_order),
                block1_conv2_b=f['/block1_conv2/block1_conv2_b_1:0'][()],

                block2_conv1_W=f['/block2_conv1/block2_conv1_W_1:0'][()].transpose(W_order),
                block2_conv1_b=f['/block2_conv1/block2_conv1_b_1:0'][()],
                block2_conv2_W=f['/block2_conv2/block2_conv2_W_1:0'][()].transpose(W_order),
                block2_conv2_b=f['/block2_conv2/block2_conv2_b_1:0'][()],

                block3_conv1_W=f['/block3_conv1/block3_conv1_W_1:0'][()].transpose(W_order),
                block3_conv1_b=f['/block3_conv1/block3_conv1_b_1:0'][()],
                block3_conv2_W=f['/block3_conv2/block3_conv2_W_1:0'][()].transpose(W_order),
                block3_conv2_b=f['/block3_conv2/block3_conv2_b_1:0'][()],
                block3_conv3_W=f['/block3_conv3/block3_conv3_W_1:0'][()].transpose(W_order),
                block3_conv3_b=f['/block3_conv3/block3_conv3_b_1:0'][()],
                block3_conv4_W=f['/block3_conv4/block3_conv4_W_1:0'][()].transpose(W_order),
                block3_conv4_b=f['/block3_conv4/block3_conv4_b_1:0'][()],

                block4_conv1_W=f['/block4_conv1/block4_conv1_W_1:0'][()].transpose(W_order),
                block4_conv1_b=f['/block4_conv1/block4_conv1_b_1:0'][()],
                block4_conv2_W=f['/block4_conv2/block4_conv2_W_1:0'][()].transpose(W_order),
                block4_conv2_b=f['/block4_conv2/block4_conv2_b_1:0'][()],
                block4_conv3_W=f['/block4_conv3/block4_conv3_W_1:0'][()].transpose(W_order),
                block4_conv3_b=f['/block4_conv3/block4_conv3_b_1:0'][()],
                block4_conv4_W=f['/block4_conv4/block4_conv4_W_1:0'][()].transpose(W_order),
                block4_conv4_b=f['/block4_conv4/block4_conv4_b_1:0'][()],

                block5_conv1_W=f['/block5_conv1/block5_conv1_W_1:0'][()].transpose(W_order),
                block5_conv1_b=f['/block5_conv1/block5_conv1_b_1:0'][()],
                block5_conv2_W=f['/block5_conv2/block5_conv2_W_1:0'][()].transpose(W_order),
                block5_conv2_b=f['/block5_conv2/block5_conv2_b_1:0'][()],
                block5_conv3_W=f['/block5_conv3/block5_conv3_W_1:0'][()].transpose(W_order),
                block5_conv3_b=f['/block5_conv3/block5_conv3_b_1:0'][()],
                block5_conv4_W=f['/block5_conv4/block5_conv4_W_1:0'][()].transpose(W_order),
                block5_conv4_b=f['/block5_conv4/block5_conv4_b_1:0'][()],
            )
        return VGG19(weights)
