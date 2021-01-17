import argparse
import math
import os
import random
import sys
import time
from typing import Iterable, Optional, Sequence
import warnings

from PIL import Image
import torch
from torch.nn.functional import mse_loss
import torch.optim as optim
from torchvision.transforms.functional import resize, to_tensor, to_pil_image

from pastiche.vgg19 import VGG19

version_txt = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_txt, 'r') as f:
    __version__ = f.read().strip()


# ************************************************************
# * Utility
# ************************************************************

def get_devices():
    devices = ['cpu']
    # As of PyTorch 1.7.0, calling torch.cuda.is_available shows a warning ("...Found no NVIDIA
    # driver on your system..."). A related issue is reported in PyTorch Issue #47038.
    # Warnings are suppressed below to prevent a warning from showing when no GPU is available.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cuda_available = torch.cuda.is_available()
    if cuda_available and torch.cuda.device_count() > 0:
        devices.append('cuda')
        for idx in range(torch.cuda.device_count()):
            devices.append('cuda:{}'.format(idx))
    return tuple(devices)

EXIT_SUCCESS = 0

# ************************************************************
# * Core
# ************************************************************

DEFAULT_CONTENT_LAYERS = ['block4_relu2']
DEFAULT_STYLE_LAYERS = ['block1_relu1', 'block2_relu1', 'block3_relu1', 'block4_relu1', 'block5_relu1']
DEFAULT_TV_WEIGHT = 0.0
# VGG details are at:
#   http://www.robots.ox.ac.uk/~vgg/research/very_deep/
# which links to:
#   https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# for the normalization details.
# > "The input images should be zero-centered by mean pixel (rather than mean image)
#    subtraction. Namely, the following BGR values should be subtracted:
#      [103.939, 116.779, 123.68]."
VGG_MEAN = [103.939, 116.779, 123.68]  # BGR means


def load_image(image_path, pixels=None, size=None):
    """
    :param pixels: Number of pixels to resize to.
    :param size: Overrides 'pixels' when present. A pair, (height, width),
                 or the maximum dimension to resize to.

    """
    x = Image.open(image_path)
    w, h = x.size  # (width, height)
    if size is not None:
        if isinstance(size, int):
            if w > h:
                h_ = int(round((size / w) * h, 0))
                w_ = size
            elif w < h:
                h_ = size
                w_ = int(round((size / h) * w, 0))
            else:
                h_ = size
                w_ = size
        else:
            h_ = size[0]
            w_ = size[1]
    elif pixels is not None:
        h_ = int(round(math.sqrt(pixels * h / w), 0))
        w_ = int(round((w / h) * h_, 0))
    else:
        h_ = h
        w_ = w
    x = resize(x, (h_, w_))
    x = to_tensor(x) * 255.0
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    # Normalize for VGG
    x[[0, 1, 2]] = x[[2, 1, 0]]  # RGB -> BGR
    for idx, shift in enumerate(VGG_MEAN):
        x[idx] -= shift
    x.unsqueeze_(0)
    return x


def save_image(input, path):
    x = input.detach().squeeze(0).to('cpu', copy=True)
    # Remove VGG normalization
    for idx, shift in enumerate(VGG_MEAN):
        x[idx] += shift
    x[[0, 1, 2]] = x[[2, 1, 0]]  # BGR -> RGB
    x.div_(255.0)
    x.clamp_(0.0, 1.0)
    x = to_pil_image(x)
    x.save(path, 'png')


def gram(input):
    _, _, h, w = input.shape
    x = input.squeeze(0)
    x = x.flatten(1)
    x = torch.mm(x, x.t())
    x.div_(h * w)
    return x


def cov(input):
    """Returns a covariance matrix for the given input matrix."""
    mu = input.mean(dim=0)
    normalized = input - mu
    x = normalized.t().mm(normalized)
    x.div_(input.shape[0])
    return x


def matrix_sqrt(input):
    eig = input.eig(eigenvectors=True)
    eig_val = eig.eigenvalues[:, 0]
    eig_vec = eig.eigenvectors
    output = eig_vec.mm(eig_val.sqrt().diag()).mm(eig_vec.t())
    return output


def transfer_color(source, target):
    """Transfers color from source to target. Doesn't clamp to valid values."""
    shape = target.shape
    # flatten the spatial dimensions
    source = source.reshape((3, -1)).t()
    target = target.reshape((3, -1)).t()
    mu_source = source.mean(dim=0)
    mu_target = target.mean(dim=0)
    cov_source = cov(source)
    cov_target = cov(target)
    x = matrix_sqrt(cov_source)
    x = x.mm(matrix_sqrt(cov_target).inverse())
    x = x.mm((target - mu_target).t()).t()
    x = x + mu_source
    output = x.t().reshape(shape)
    return output


class PasticheArtist:
    def __init__(
            self,
            vgg19: VGG19,
            init: torch.Tensor,
            content: torch.Tensor,
            style: torch.Tensor,
            content_layers: Iterable=DEFAULT_CONTENT_LAYERS,
            style_layers: Iterable=DEFAULT_STYLE_LAYERS,
            content_weights: Optional[Sequence]=None,
            style_weights: Optional[Sequence]=None,
            tv_weight: float=DEFAULT_TV_WEIGHT):
        self.vgg19 = vgg19
        self.pastiche = init.clone().requires_grad_()
        self.content = content
        self.style = style
        self.optimizer = optim.LBFGS([self.pastiche], max_iter=1)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_targets = self.vgg19.forward(self.content, self.content_layers)
        self.style_targets = self.vgg19.forward(self.style, self.style_layers)
        self.content_weights = content_weights
        self.style_weights = style_weights
        self.tv_weight = tv_weight
        self.loss = self._calc_loss().item()

    def _calc_loss(self):
        loss = torch.tensor(0.0, requires_grad=True, device=self.vgg19.block1_conv1.weight.device)

        pastiche_layers = list(self.content_layers) + list(self.style_layers)
        pastiche_targets = self.vgg19.forward(self.pastiche, pastiche_layers)

        # content loss
        for idx, layer in enumerate(self.content_layers):
            pastiche_act = pastiche_targets[layer]
            content_act = self.content_targets[layer]
            if self.content_weights is None or len(self.content_weights) == 0:
                weight = 1.0
            elif len(self.content_weights) > idx:
                weight = self.content_weights[idx]
            else:
                weight = self.content_weights[-1]
            loss = loss + weight * mse_loss(pastiche_act, content_act)

        # style loss
        for idx, layer in enumerate(self.style_layers):
            pastiche_act = pastiche_targets[layer]
            style_act = self.style_targets[layer]
            pastiche_g = gram(pastiche_act)
            style_g = gram(style_act)
            if self.style_weights is None or len(self.style_weights) == 0:
                weight = 1e3 / (pastiche_act.shape[1] ** 2)
            elif len(self.style_weights) > idx:
                weight = self.style_weights[idx]
            else:
                weight = self.style_weights[-1]
            loss = loss + weight * mse_loss(pastiche_g, style_g.detach())

        # total-variation loss
        tv_loss = (self.pastiche[:,:,:,1:] - self.pastiche[:,:,:,:-1]).abs().sum()
        tv_loss += (self.pastiche[:,:,1:,:] - self.pastiche[:,:,:-1,:]).abs().sum()
        loss = loss + self.tv_weight * tv_loss

        return loss

    def draw(self):
        def closure():
            self.optimizer.zero_grad()
            loss = self._calc_loss()
            loss.backward()
            self.loss = loss.item()
            return loss

        self.optimizer.step(closure)


# ************************************************************
# * Command Line Interface
# ************************************************************

class _ListLayersAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        kwargs = {
            'option_strings': option_strings,
            'dest': dest,
            'default': default,
            'nargs': 0,
        }
        if help is not None:
            kwargs['help'] = help
        super(_ListLayersAction, self).__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        idx_padding = len(str(len(VGG19.LAYER_NAMES)))
        for idx, name in enumerate(VGG19.LAYER_NAMES):
            if idx > 0 and name[:6] != VGG19.LAYER_NAMES[idx - 1][:6]:
                print()
            print(f'{idx + 1: >{idx_padding}}. {name}')
        parser.exit()


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='pastiche',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    devices = get_devices()

    # General options
    parser.add_argument('--version',
                        action='version',
                        version='pastiche {}'.format(__version__))
    parser.add_argument('--device', default='cuda' if 'cuda' in devices else 'cpu', choices=devices)
    parser.add_argument('--seed', type=int, help='RNG seed.')
    parser.add_argument(
        '--deterministic', action='store_true', help='Avoid non-determinism where possible (at cost of speed).')
    parser.add_argument(
        '--list-layers', action=_ListLayersAction, help='Show a list of available layer names and exit.')
    # Optimization options
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument(
        '--content-layers',
        choices=VGG19.LAYER_NAMES,
        metavar='LAYER_NAME',
        nargs='*',
        default=DEFAULT_CONTENT_LAYERS,
        help='Content layer names. Use --list-layers to show a list of choices.')
    parser.add_argument(
        '--style-layers',
        choices=VGG19.LAYER_NAMES,
        metavar='LAYER_NAME',
        nargs='*',
        default=DEFAULT_STYLE_LAYERS,
        help='Style layer names. Use --list-layers to show a list of choices.')
    parser.add_argument('--content-weights', metavar='WEIGHT', nargs='*', type=float)
    parser.add_argument('--style-weights', metavar='WEIGHT', nargs='*', type=float)
    parser.add_argument('--tv-weight', default=DEFAULT_TV_WEIGHT, type=float, help='Total-variation weight')
    # Output options
    parser.add_argument('--no-verbose', action='store_false', dest='verbose')
    parser.add_argument('--info-step', type=int, default=100, help='Step size for displaying information.')
    parser.add_argument('--workspace', help='Directory for saving intermediate results.')
    parser.add_argument('--workspace-step', type=int, default=10, help='Step size for saving to workspace.')
    # Input options
    parser.add_argument('--random-init', action='store_true', help='Initialize randomly (overrides --init)')
    parser.add_argument('--init', help='Optional file path to the initialization image.')
    parser.add_argument(
        '--size-pixels',
        type=int,
        default=500 ** 2,  # recommendation from "Controlling Perceptual Factors in Neural Style Transfer"
        help='Approximate number of pixels for content and pastiche images.')
    parser.add_argument(
        '--size',
        type=int,
        help='Maximum dimension for content and pastiche images. Overrides --size-pixels when present.')
    parser.add_argument(
        '--style-size-pixels',
        type=int,
        help='Approximate number of pixels for style image.')
    parser.add_argument(
        '--style-size',
        type=int,
        help='Maximum dimension for style image. Overrides --style-size-pixels when present.')
    parser.add_argument('--preserve-color', action='store_true', help='Preserve color of content image.')
    # Required options
    parser.add_argument('content', help='File path to the content image.')
    parser.add_argument('style', help='File path to the style image.')
    parser.add_argument('output', help='File path to save the PNG image.')

    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    start_time = time.time()
    args = _parse_args(argv)
    if not args.output.lower().endswith('.png'):
        sys.stderr.write('Output file is missing PNG extension.\n')
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    vgg19_q_bin_path = os.path.join(
        os.path.dirname(__file__), 'vgg19_weights_tf_dim_ordering_tf_kernels_notop_q.bin')
    vgg19 = VGG19.from_quantized_bin(vgg19_q_bin_path).to(args.device)
    content = load_image(args.content, pixels=args.size_pixels, size=args.size).to(args.device)
    style_pixels = args.style_size_pixels
    if style_pixels is None:
        style_pixels = content.shape[2] * content.shape[3]
    style = load_image(args.style, pixels=style_pixels, size=args.style_size).to(args.device)
    if args.preserve_color:
        style.data = transfer_color(content, style)
    if args.random_init:
        init = torch.randn(content.shape, device='cuda')
    elif args.init is not None:
        init = load_image(args.init, size=content.shape[2:]).to(args.device)
    else:
        init = content.clone()

    artist = PasticheArtist(
        vgg19, init, content, style,
        content_layers=args.content_layers,
        style_layers=args.style_layers,
        content_weights=args.content_weights,
        style_weights=args.style_weights,
        tv_weight=args.tv_weight)

    # The 0th step does nothing, which is why there are (args.num_steps + 1) total steps
    max_step_str_width = len(str(args.num_steps))
    if args.verbose:
        print(f'device: {args.device}')
        print(f'seed: {seed}')
        print(f'size: {"x".join(str(x) for x in reversed(content.shape[2:]))}')
        print(f'style_size: {"x".join(str(x) for x in reversed(style.shape[2:]))}')
        print()
        print('step elapsed loss')
        print('---- ------- ----')
    for step in range(args.num_steps + 1):
        if step > 0:
            artist.draw()
        if args.workspace is not None and step % args.workspace_step == 0:
            os.makedirs(args.workspace, exist_ok=True)
            name = f'{step:0{max_step_str_width}d}.png'
            path = os.path.join(args.workspace, name)
            save_image(artist.pastiche, path)
        if args.verbose and step % args.info_step == 0:
            elapsed = time.time() - start_time
            info = f'{step: <4} {elapsed: <7.1f} {artist.loss:.2f}'
            print(info)
    save_image(artist.pastiche, args.output)

    return EXIT_SUCCESS


if __name__ == '__main__':
    sys.exit(main(sys.argv))
