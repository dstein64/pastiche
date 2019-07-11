"""
This script generates 8 new versions of vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5,
each with quantized weights. A suffix is added to the filename indicating the level of
quantization (how many bits would be required to represent each weight).
"""

import os
import shutil

import h5py
import kmeans1d
import numpy as np

vgg19_h5_path = os.path.join(
        os.path.dirname(__file__), 'pastiche', 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

BITS = [8, 7, 6, 5, 4, 3, 2, 1]

QUANTIZE_LAYERS = [
    '/block1_conv1/block1_conv1_W_1:0',
    '/block1_conv2/block1_conv2_W_1:0',
    '/block2_conv1/block2_conv1_W_1:0',
    '/block2_conv2/block2_conv2_W_1:0',
    '/block3_conv1/block3_conv1_W_1:0',
    '/block3_conv2/block3_conv2_W_1:0',
    '/block3_conv3/block3_conv3_W_1:0',
    '/block3_conv4/block3_conv4_W_1:0',
    '/block4_conv1/block4_conv1_W_1:0',
    '/block4_conv2/block4_conv2_W_1:0',
    '/block4_conv3/block4_conv3_W_1:0',
    '/block4_conv4/block4_conv4_W_1:0',
    '/block5_conv1/block5_conv1_W_1:0',
    '/block5_conv2/block5_conv2_W_1:0',
    '/block5_conv3/block5_conv3_W_1:0',
    '/block5_conv4/block5_conv4_W_1:0',
]

print('bits layer')
for b in BITS:
    k = 2 ** b
    target_path = os.path.join(
        os.path.dirname(__file__), 'pastiche', f'vgg19_weights_tf_dim_ordering_tf_kernels_notop_q{b}.h5')
    shutil.copyfile(vgg19_h5_path, target_path)

    with h5py.File(target_path, 'r+') as f:
        for layer in QUANTIZE_LAYERS:
            print(f'{b} {layer}')
            shape = f[layer][...].shape
            data = f[layer][...].flatten()
            clusters, centroids = kmeans1d.cluster(data, k)
            f[layer][...] = np.array([centroids[cluster] for cluster in clusters]).reshape(shape)
