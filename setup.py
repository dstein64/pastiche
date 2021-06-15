import os
from setuptools import setup

version_txt = os.path.join(os.path.dirname(__file__), 'pastiche', 'version.txt')
with open(version_txt, 'r') as f:
    version = f.read().strip()

with open('README.md') as f:
    long_description = f.read()

setup(
    author='Daniel Steinberg',
    author_email='ds@dannyadam.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Artistic Software',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
    ],
    description='A PyTorch implementation of Neural Style Transfer (NST)',
    entry_points={
        'console_scripts': ['pastiche=pastiche.pastiche:main'],
    },
    extras_require={'dev': ['kmeans1d']},
    keywords=['gatys', 'style-transfer', 'neural-style-transfer'],
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    name='pastiche',
    package_data={'pastiche': ['version.txt', 'vgg19_weights_tf_dim_ordering_tf_kernels_notop_q.bin']},
    packages=['pastiche'],
    python_requires='>=3.6',
    install_requires=['h5py', 'numpy', 'pillow', 'torch', 'torchvision'],
    url='https://github.com/dstein64/pastiche',
    version=version,
)
