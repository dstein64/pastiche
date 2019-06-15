import argparse
import os
import random
import sys
from typing import Iterable, Optional, Sequence

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import resize, to_tensor, to_pil_image

from pastiche.vgg19 import VGG19

version_txt = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_txt, 'r') as f:
    __version__ = f.read().strip()


# ************************************************************
# * Utility
# ************************************************************

DEVICES = ['cpu']
if torch.cuda.torch.cuda.is_available() and torch.cuda.device_count() > 0:
    DEVICES.append('cuda')
    for idx in range(torch.cuda.device_count()):
        DEVICES.append('cuda:{}'.format(idx))
DEVICES = tuple(DEVICES)


# ************************************************************
# * Core
# ************************************************************

DEFAULT_CONTENT_LAYERS = ['block4_relu2']
DEFAULT_STYLE_LAYERS = ['block1_relu1', 'block2_relu1', 'block3_relu1', 'block4_relu1', 'block5_relu1']


def load_image(image_name, size=None):
    image = Image.open(image_name)
    if size is not None:
        image = resize(image, size)
    image = to_tensor(image) * 255.0
    image = image.unsqueeze(0)
    return image


def save_image(input, path):
    x = input.detach().squeeze(0).div(255.0).clamp(0.0, 1.0).to('cpu')
    x = to_pil_image(x)
    x.save(path, 'png')


def gram(input):
    _, _, h, w = input.shape
    x = input.squeeze(0)
    x = x.flatten(1)
    x = torch.mm(x, x.t())
    x = x.div(h * w)
    return x


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
            style_weights: Optional[Sequence]=None):
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
        self.loss = self.calc_loss().item()

    def calc_loss(self):
        loss = torch.tensor(0.0, requires_grad=True, device=self.vgg19.block1_conv1.weight.device)

        pastiche_layers = list(self.content_layers) + list(self.style_layers)
        pastiche_targets = self.vgg19.forward(self.pastiche, pastiche_layers)
        for idx, layer in enumerate(self.content_layers):
            pastiche_act = pastiche_targets[layer]
            content_act = self.content_targets[layer]
            if self.content_weights is None or len(self.content_weights) == 0:
                weight = 1.0
            elif len(self.content_weights) > idx:
                weight = self.content_weights[idx]
            else:
                weight = self.content_weights[-1]
            loss = loss + weight * nn.MSELoss()(pastiche_act, content_act)

        for idx, layer in enumerate(self.style_layers):
            pastiche_act = pastiche_targets[layer]
            style_act = self.style_targets[layer]
            pastiche_g = gram(pastiche_act)
            style_g = gram(style_act)
            if self.style_weights is None or len(self.style_weights) == 0:
                weight = 2e3 / (pastiche_act.shape[1] ** 2)
            elif len(self.style_weights) > idx:
                weight = self.style_weights[idx]
            else:
                weight = self.style_weights[-1]
            loss = loss + weight * nn.MSELoss()(pastiche_g, style_g.detach())

        return loss


    def draw(self):
        def closure():
            self.optimizer.zero_grad()
            loss = self.calc_loss()
            loss.backward()
            self.loss = loss.item()
            return loss

        self.optimizer.step(closure)


# ************************************************************
# * Command Line Interface
# ************************************************************

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='pastiche',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--version',
                        action='version',
                        version='pastiche {}'.format(__version__))
    parser.add_argument('--seed', type=int, help='RNG seed.')
    parser.add_argument('--device', default='cuda' if 'cuda' in DEVICES else 'cpu', choices=DEVICES)
    parser.add_argument('--no-verbose', action='store_false', dest='verbose')
    parser.add_argument('--num-steps', type=int, default=5000)
    parser.add_argument('--info-step', type=int, default=100, help='Step size for displaying information.')
    parser.add_argument('--workspace', help='Directory for saving intermediate results.')
    parser.add_argument('--workspace-step', type=int, default=10, help='Step size for saving to workspace.')
    parser.add_argument('--random-init', action='store_true', help='Initialize randomly (overrides --init)')
    parser.add_argument('--init', help='Optional file path to the initialization image.')
    parser.add_argument(
        '--content-layers', choices=VGG19.LAYER_NAMES, nargs='*', default=DEFAULT_CONTENT_LAYERS)
    parser.add_argument(
        '--style-layers', choices=VGG19.LAYER_NAMES, nargs='*', default=DEFAULT_STYLE_LAYERS)
    parser.add_argument('--content-weights', nargs='*', type=float)
    parser.add_argument('--style-weights', nargs='*', type=float)
    parser.add_argument('content', help='File path to the content image.')
    parser.add_argument('style', help='File path to the style image.')
    parser.add_argument('output', help='File path to save the PNG image.')
    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    args = _parse_args(argv)
    if not args.output.lower().endswith('.png'):
        sys.stderr.write('Output file is missing PNG extension.\n')
    if args.verbose:
        print('Device: {}'.format(args.device))
    # TODO: rand handling
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    if args.verbose:
        print('Seed: {}'.format(seed))

    vgg19_h5_path = os.path.join(
        os.path.dirname(__file__), 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg19 = VGG19.from_h5(vgg19_h5_path).to(args.device)
    content = load_image(args.content, size=512).to(args.device) # TODO: real size
    style = load_image(args.style, size=512 * 2).to(args.device) # TODO: real size
    if args.random_init:
        # Even though actual image is comprised of pixels with intensities between 0 and 255,
        # initializing from a standard normal works well. Negatives are clamped later, but the
        # optimization procedure pushes intensities to be positive.
        init = torch.randn(content.shape, device='cuda')
    elif args.init is not None:
        # TODO: make sure that size= supports 2D arg
        init = load_image(args.init, size=content.shape[2:]).to(args.device)
    else:
        init = content.clone()

    artist = PasticheArtist(
        vgg19, init, content, style,
        content_layers=args.content_layers,
        style_layers=args.style_layers,
        content_weights=args.content_weights,
        style_weights=args.style_weights)

    # The 0th step does nothing, which is why there are args.num_steps + 1 total steps
    padding_width = len(str(args.num_steps))
    for step in range(args.num_steps + 1):
        if step > 0:
            artist.draw()
        if args.workspace is not None and step % args.workspace_step == 0:
            name = f'{step:0{padding_width}d}.png'
            path = os.path.join(args.workspace, name)
            save_image(artist.pastiche, path)
        if step % args.info_step == 0:
            info = f'{step: <{padding_width}d} {artist.loss:.2f}'
            print(info)
    save_image(artist.pastiche, args.output)

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(sys.argv))
