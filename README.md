pastiche
========

A PyTorch-based Python implementation of Neural Style Transfer (Gatys et al. 2015).

Features
--------

- Support for saving intermediate images during optimization

Installation
------------

#### Requirements

- Python 3.5 or greater

#### Install

```sh
$ pip3 install pastiche
```

#### Update

```sh
$ pip3 install --upgrade pastiche
```

Usage
-----

The program is intended to be used from the command line.

There are various options, including but not limited to:
- Device (CPU versus GPU)
- Number of optimization iterations
- VGG layers to utilize
- Loss function term weights

For the full list of options and the corresponding documentation, see the source code or use `--help`.

```sh
$ pastiche --help
```

Example Usage
-------------

This examples applies the style from Vincent van Gogh's `The Starry Night` to a photo I took in Boston
in 2015.

```sh
$ pastiche \
    --num-steps 5000 \
    --size 1024 \
    --device cuda \
    boston.jpg \
    vangogh_starry_night.jpg \
    pastiche.png
```

| pastiche.png |      
|:----------:|
| <img src="https://github.com/dstein64/pastiche/blob/master/example/pastiche.png?raw=true" width="600"/> |

| vangogh_starry_night.jpg | boston.jpg |
|:------------------------:|:----------:|
| <img src="https://github.com/dstein64/pastiche/blob/master/example/vangogh_starry_night.jpg?raw=true" width="300"/> | <img src="https://github.com/dstein64/pastiche/blob/master/example/boston.jpg?raw=true" width="300"/> |


License
-------

The source code has an [MIT License](https://en.wikipedia.org/wiki/MIT_License).

See [LICENSE](https://github.com/dstein64/pastiche/blob/master/LICENSE).

References
----------

Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A Neural Algorithm of Artistic Style."
ArXiv:1508.06576 [Cs, q-Bio], August 26, 2015. http://arxiv.org/abs/1508.06576.
