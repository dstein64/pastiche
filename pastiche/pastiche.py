import argparse
from collections import defaultdict
from itertools import groupby
import math
import os
import random
import sys
import time
from typing import Iterable, Optional, Sequence, List
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
    if x.shape[0] == 2:
        raise RuntimeError('Unsupported image format.')
    elif x.shape[0] == 1:
        # Convert monochrome image to RGB
        x = x.repeat(3, 1, 1)
    elif x.shape[0] > 3:
        # Drop alpha channel
        x = x[:3]
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
            content: str,
            styles: str,
            *,
            pooling: str = 'max',
            optimizer: str = 'lbfgs',
            random_init: bool = False,
            init: Optional[str] = None,
            device: str = 'cuda' if 'cuda' in get_devices() else 'cpu',
            supplemental_devices: Optional[List[List]] = None,
            preserve_color: bool = False,
            content_layers: Iterable = DEFAULT_CONTENT_LAYERS,
            style_layers: Iterable = DEFAULT_STYLE_LAYERS,
            content_layer_weights: Optional[Sequence] = None,
            style_layer_weights: Optional[Sequence] = None,
            content_weight: float = 1.0,
            style_weights: Optional[Sequence] = None,
            tv_weight: float = 0.0,
            size_pixels: Optional[int] = None,
            size=None,
            style_size_pixels: Optional[int] = None,
            style_size=None):
        if supplemental_devices is None:
            supplemental_devices = []
        # Configure the device strategy.
        # 'device_strategy' maps layer indices to devices
        self.device_strategy = [None] * len(VGG19.LAYER_NAMES)
        pastiche_layers = set(content_layers).union(style_layers)
        end = max(VGG19.LAYER_INDEX_LOOKUP[layer] for layer in pastiche_layers) + 1
        for supp_device, start in sorted(supplemental_devices, key=lambda x: x[1], reverse=True):
            for idx in range(start, end):
                self.device_strategy[idx] = supp_device
            end = start
        for idx in range(0, end):
            self.device_strategy[idx] = device
        vgg19_q_bin_path = os.path.join(
            os.path.dirname(__file__), 'vgg19_weights_tf_dim_ordering_tf_kernels_notop_q.bin')
        vgg19 = VGG19.from_quantized_bin(vgg19_q_bin_path, pooling=pooling)
        vgg19.set_device_strategy(self.device_strategy)
        # Disable gradient calculations for model weights to reduce memory requirement and reduce runtime.
        for param in vgg19.parameters():
            param.requires_grad = False
        content = load_image(content, pixels=size_pixels, size=size)
        self.content_size = list(content.shape[2:])
        content_targets = vgg19.forward(content.to(device), content_layers)
        style_pixels = style_size_pixels
        if style_pixels is None:
            style_pixels = self.content_size[0] * self.content_size[1]
        style_targetss = []
        self.style_sizes = []
        for path in styles:
            style = load_image(path, pixels=style_pixels, size=style_size)
            self.style_sizes.append(list(style.shape[2:]))
            if preserve_color:
                style.data = transfer_color(content, style)
            style_targetss.append(vgg19.forward(style.to(device), style_layers))
            del style
        if random_init:
            self.pastiche = torch.randn(content.shape, device=device)
        elif init is not None:
            self.pastiche = load_image(init, size=self.content_size).to(device)
        else:
            self.pastiche = content.to(device, copy=True)
        self.pastiche.requires_grad_()
        del content

        self.vgg19 = vgg19
        if optimizer == 'lbfgs':
            self.optimizer = optim.LBFGS([self.pastiche], max_iter=1, lr=1.0)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam([self.pastiche], lr=1.0)
        else:
            raise RuntimeError(f'Unknown optimizer: {optimizer}')
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_targets = content_targets
        self.style_targetss = style_targetss
        self.content_layer_weights = content_layer_weights
        self.style_layer_weights = style_layer_weights
        self.content_weight = content_weight
        self.style_weights = style_weights
        if self.style_weights is None:
            self.style_weights = []
        self.tv_weight = tv_weight
        self.loss = self._calc_loss().item()

    def device_layers_reprs(self):
        """Returns a dictionary mapping devices to a string representation of assigned layers."""
        mapping = defaultdict(list)
        idx = 0
        for device, grouper in groupby(self.device_strategy):
            if device is None:
                continue
            num_layers = len(list(grouper))
            repr_ = str(idx)
            if num_layers > 1:
                repr_ = f'{repr_}-{idx + num_layers - 1}'
            mapping[device].append(repr_)
            idx += num_layers
        output = {key: ','.join(value) for key, value in mapping.items()}
        return output

    def _calc_loss(self):
        # Using the CPU for loss calculations does not have a substantive impact when using GPU
        # elsewhere, and accommodates multi-device scenarios without added complexity.
        loss = torch.tensor(0.0, requires_grad=True, device='cpu')

        pastiche_layers = list(self.content_layers) + list(self.style_layers)
        pastiche_targets = self.vgg19.forward(self.pastiche, pastiche_layers)

        # content loss
        for idx, layer in enumerate(self.content_layers):
            pastiche_act = pastiche_targets[layer]
            content_act = self.content_targets[layer]
            if self.content_layer_weights is None or len(self.content_layer_weights) == 0:
                layer_weight = .01
            elif len(self.content_layer_weights) > idx:
                layer_weight = self.content_layer_weights[idx]
            else:
                layer_weight = self.content_layer_weights[-1]
            layer_loss = layer_weight * mse_loss(pastiche_act, content_act).to('cpu')
            loss = loss + self.content_weight * layer_loss

        # style loss
        for style_idx, style_targets in enumerate(self.style_targetss):
            style_weight = 1.0 / len(self.style_targetss)
            if style_idx < len(self.style_weights):
                style_weight = self.style_weights[style_idx]
            for layer_idx, layer in enumerate(self.style_layers):
                pastiche_act = pastiche_targets[layer]
                style_act = style_targets[layer]
                pastiche_g = gram(pastiche_act)
                style_g = gram(style_act)
                if self.style_layer_weights is None or len(self.style_layer_weights) == 0:
                    layer_weight = 10.0 / (pastiche_act.shape[1] ** 2)
                elif len(self.style_layer_weights) > layer_idx:
                    layer_weight = self.style_layer_weights[layer_idx]
                else:
                    layer_weight = self.style_layer_weights[-1]
                layer_loss = layer_weight * mse_loss(pastiche_g, style_g.detach()).to('cpu')
                loss = loss + style_weight * layer_loss

        # total-variation loss
        tv_loss = (self.pastiche[:, :, :, 1:] - self.pastiche[:, :, :, :-1]).abs().sum().to('cpu')
        tv_loss += (self.pastiche[:, :, 1:, :] - self.pastiche[:, :, :-1, :]).abs().sum().to('cpu')
        loss = loss + self.tv_weight * tv_loss

        return loss

    def draw(self):
        assert self.pastiche.requires_grad

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
        idx_padding = len(str(len(VGG19.LAYER_NAMES) - 1))
        for idx, name in enumerate(VGG19.LAYER_NAMES):
            if idx > 0 and name[:6] != VGG19.LAYER_NAMES[idx - 1][:6]:
                print()
            print(f'{idx: >{idx_padding}}. {name}')
        parser.exit()


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='pastiche',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    devices = get_devices()
    last_layer_idx = len(VGG19.LAYER_NAMES) - 1

    # General options
    parser.add_argument(
        '--version',
        action='version',
        version='pastiche {}'.format(__version__)
    )
    parser.add_argument(
        '--device', '-d',
        help='Primary device to use for computations. Additional devices can be specified with --supplemental-device.',
        default='cuda' if 'cuda' in devices else 'cpu',
        choices=devices
    )
    parser.add_argument(
        '--supplemental-device',
        nargs=2,
        metavar=('DEVICE', 'INDEX'),
        action='append',
        dest='supplemental_devices',
        help='Supplemental device to use for computations along with a layer index specifying the first layer for'
             ' which the device will be used. The --supplemental-device argument can be repeated. For example,'
             ' "--device cuda:0 --supplemental-device cuda:1 10 --supplemental-device cpu 20" configures GPU 0 for'
             f' layers 0 through 9, GPU 1 for layers 10 through 19, and the CPU for layers 20 through {last_layer_idx}.'
             f' Available devices: {", ".join(devices)}. Available layer indices: 1 through {last_layer_idx}.'
    )
    parser.add_argument('--seed', type=int, help='RNG seed.')
    parser.add_argument(
        '--deterministic', action='store_true', help='Avoid non-determinism where possible (at cost of speed).')
    parser.add_argument(
        '--list-layers', action=_ListLayersAction, help='Show a list of available layer names and exit.')

    # VGG model options
    parser.add_argument('--pooling', choices=('max', 'avg'), default='max')

    # Optimization options
    parser.add_argument('--optimizer', choices=('adam', 'lbfgs'), default='lbfgs')
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument(
        '--content-layers',
        choices=VGG19.LAYER_NAMES,
        metavar='LAYER_NAME',
        nargs='*',
        default=DEFAULT_CONTENT_LAYERS,
        help='Content layer names. Use --list-layers to show a list of choices.'
    )
    parser.add_argument(
        '--style-layers',
        choices=VGG19.LAYER_NAMES,
        metavar='LAYER_NAME',
        nargs='*',
        default=DEFAULT_STYLE_LAYERS,
        help='Style layer names. Use --list-layers to show a list of choices.'
    )
    parser.add_argument(
        '--content-layer-weights',
        metavar='WEIGHT',
        nargs='*',
        type=float,
        help='Weights corresponding to --content-layers.'
    )
    parser.add_argument(
        '--style-layer-weights',
        metavar='WEIGHT',
        nargs='*',
        type=float,
        help='Weights corresponding to --style-layers.'
    )
    parser.add_argument(
        '--content-weight',
        type=float,
        default=1.0,
        help='Content image weighting.'
    )
    parser.add_argument(
        '--style-weights',
        type=float,
        metavar='STYLE_WEIGHT',
        nargs='*',
        help='Style image(s) weighting. Defaults to 1/N for each of N style images.'
    )
    parser.add_argument('--tv-weight', default=0.0, type=float, help='Total-variation weight')

    # Output options
    parser.add_argument('--no-verbose', action='store_false', dest='verbose')
    parser.add_argument('--info-step', type=int, default=100, help='Step size for displaying information.')
    parser.add_argument('--workspace', help='Directory for saving intermediate results.')
    parser.add_argument('--workspace-prefix', help='Prefix for workspace images.', default='')
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
        help='Approximate number of pixels for style image(s).')
    parser.add_argument(
        '--style-size',
        type=int,
        help='Maximum dimension for style image(s). Overrides --style-size-pixels when present.')
    parser.add_argument('--preserve-color', action='store_true', help='Preserve color of content image.')
    # Required options
    # Flags are used instead of requiring positional arguments to prevent ambiguity when such arguments are
    # preceded by an argument with a variable number of inputs (e.g., nargs='*'). This was deemed preferable
    # to requiring a '--' separator to resolve such ambiguities.
    parser.add_argument('--content', '-c', required=True, help='File path to the content image.')
    parser.add_argument(
        '--styles', '-s', metavar='STYLE', nargs='+', required=True, help='File path(s) to the style image(s).')
    parser.add_argument('--output', '-o', required=True, help='File path to save the PNG image.')

    args = parser.parse_args(argv[1:])
    if args.supplemental_devices is None:
        args.supplemental_devices = []
    if args.style_weights is None:
        args.style_weights = []
    if len(args.style_weights) < len(args.styles):
        args.style_weights.append(1.0 / len(args.styles))

    if not args.output.lower().endswith('.png'):
        sys.stderr.write('Output file is missing PNG extension.\n')
        # Intentionally no exit after warning, as processing can proceed even though the output
        # file doesn't have the expected extension. The saved file contents will be a PNG image.
    for supplemental_device in args.supplemental_devices:
        try:
            supplemental_device[1] = int(supplemental_device[1])
        except ValueError:
            sys.stderr.write(f'Invalid --supplemental-device layer index: {supplemental_device[1]}\n')
            sys.exit(1)
    for device, idx in args.supplemental_devices:
        if device not in devices:
            sys.stderr.write(f'Invalid --supplemental-device device: {device}\n')
            sys.exit(1)
        if idx <= 0 or idx > last_layer_idx:
            sys.stderr.write(f'Invalid --supplemental-device layer index: {idx}\n')
            sys.exit(1)
    supp_device_layer_indices = sorted([idx for _, idx in args.supplemental_devices])
    for idx1, idx2 in zip(supp_device_layer_indices, supp_device_layer_indices[1:]):
        if idx1 == idx2:
            sys.stderr.write(f'Repeated --supplemental-device layer index: {idx1}\n')
            sys.exit(1)

    return args


def main(argv=sys.argv):
    start_time = time.time()
    args = _parse_args(argv)
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    artist = PasticheArtist(
        args.content, args.styles,
        pooling=args.pooling,
        optimizer=args.optimizer,
        device=args.device,
        supplemental_devices=args.supplemental_devices,
        random_init=args.random_init,
        init=args.init,
        preserve_color=args.preserve_color,
        content_layers=args.content_layers,
        style_layers=args.style_layers,
        content_layer_weights=args.content_layer_weights,
        style_layer_weights=args.style_layer_weights,
        content_weight=args.content_weight,
        style_weights=args.style_weights,
        tv_weight=args.tv_weight,
        size_pixels=args.size_pixels,
        size=args.size,
        style_size_pixels=args.style_size_pixels,
        style_size=args.style_size
    )

    # The 0th step does nothing, which is why there are (args.num_steps + 1) total steps
    max_step_str_width = len(str(args.num_steps))
    if args.verbose:
        for idx, (device, layers) in enumerate(artist.device_layers_reprs().items()):
            print(f'device[{idx}]: {device} ({layers})')
        print(f'seed: {seed}')
        print(f'size: {"x".join(str(x) for x in reversed(artist.content_size))}')
        for idx in range(len(args.styles)):
            print(f'style_sizes[{idx}]: {"x".join(str(x) for x in reversed(artist.style_sizes[idx]))}')
        print()
        print('step elapsed loss')
        print('---- ------- ----')
    for step in range(args.num_steps + 1):
        if step > 0:
            artist.draw()
        if args.workspace is not None and step % args.workspace_step == 0:
            os.makedirs(args.workspace, exist_ok=True)
            name = f'{args.workspace_prefix}{step:0{max_step_str_width}d}.png'
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
