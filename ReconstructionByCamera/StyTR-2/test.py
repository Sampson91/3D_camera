import argparse
from pathlib import Path
import os
import torch
import torch.nn as torch_neural_network
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calculate_mean_standard, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time
from tqdm import tqdm


def test_transform(size, crop):
    transform_list = []

    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transform(height, width):
    height_width = (height, width)
    size = int(np.max(height_width))
    transform_list = []
    transform_list.append(transforms.CenterCrop((height, width)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def content_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_directory', default='./datasets/test_content',
                    type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', default='./datasets/test_style', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg', type=str,
                    default='./experiments/vgg_normalized.pth')
parser.add_argument('--decoder_path', type=str,
                    default='experiments/decoder_iteration_160000.pth')
parser.add_argument('--Transformers_path', type=str,
                    default='experiments/transformer_iteration_160000.pth')
parser.add_argument('--embedding_path', type=str,
                    default='experiments/embedding_iteration_160000.pth')

parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--position_embedding', default='learned', type=str,
                    choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dimension', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()

# Advanced options
content_size = 512
style_size = content_size
crop = content_size  # 'store_true'
save_ext = '.jpg'
output_path = args.output
preserve_color = 'store_true'
alpha = args.alpha

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Either --content or --content_dir should be given.
if args.content:
    content_paths = [Path(args.content)]
else:
    content_directory = Path(args.content_directory)
    content_paths = [file_ for file_ in content_directory.glob('*')]

# Either --style or --style_dir should be given.
if args.style:
    style_paths = [Path(args.style)]
else:
    style_directory = Path(args.style_dir)
    style_paths = [file_ for file_ in style_directory.glob('*')]

if not os.path.exists(output_path):
    os.mkdir(output_path)

vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = torch_neural_network.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()
Transformers = transformer.Transformer()

decoder.eval()
Transformers.eval()
vgg.eval()
embedding.eval()

from collections import OrderedDict

decoder_state_dict = OrderedDict()
state_dictionary = torch.load(args.decoder_path)
for key_, value_ in state_dictionary.items():
    # namekey = k[7:] # remove `module.`
    namekey = key_
    decoder_state_dict[namekey] = value_
decoder.load_state_dict(decoder_state_dict, strict=True)

Trans_state_dict = OrderedDict()
state_dictionary = torch.load(args.Transformers_path)
for key_, value_ in state_dictionary.items():
    # namekey = k[7:] # remove `module.`
    namekey = key_
    Trans_state_dict[namekey] = value_
Transformers.load_state_dict(Trans_state_dict, strict=True)

embedding_state_dict = OrderedDict()
state_dictionary = torch.load(args.embedding_path)
for key_, value_ in state_dictionary.items():
    # namekey = k[7:] # remove `module.`
    namekey = key_
    embedding_state_dict[namekey] = value_
embedding.load_state_dict(embedding_state_dict, strict=True)

with torch.no_grad():
    network = StyTR.StyTrans(vgg, decoder, embedding, Transformers, args, evaluation=True)

network.train()
network.to(device)

content_transformer = test_transform(content_size, crop)
style_transformer = test_transform(style_size, crop)

for content_path in content_paths:
    for style_path in style_paths:
        # content_tf1 = content_transform()

        content = content_transformer(Image.open(content_path).convert("RGB"))

        # c, h, w = np.shape(content)
        # style_tf1 = style_transform(h, w)
        style = style_transformer(Image.open(style_path).convert("RGB"))

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        # with torch.no_grad():
        #     output, loss_content, loss_style, l_identity1, l_identity2 = network(
        #         content, style)
        with torch.no_grad():
            output = network(content, style)

        output = output.cpu()

        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            output_path, splitext(basename(content_path))[0],
            splitext(basename(style_path))[0], save_ext
        )

        save_image(output, output_name)
        print('image saved to ->', output_name)
