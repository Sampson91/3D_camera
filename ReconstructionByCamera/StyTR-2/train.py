import argparse
import os

import numpy as np
import torch
import torch.nn as torch_neural_network
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import models.transformer as transformer
import models.StyTR as StyTR
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
import save_loss2txt


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),  ## default resize 512
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(
                        os.path.join(self.root, file_name)):
                    self.paths.append(
                        self.root + "/" + file_name + "/" + file_name1)
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        iamges = Image.open(str(path)).convert('RGB')
        iamges = self.transform(iamges)
        return iamges

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count,
                         warmup_final_learning_rate):
    """Imitating the original implementation"""
    learning_rate = warmup_final_learning_rate / (
            1.0 + arguments.learning_rate_decay * (iteration_count - 1e4))  # 2e-4
    for parameter_group_ in optimizer.param_groups:
        parameter_group_['lr'] = learning_rate
    # lr cannot be fully named otherwise, no error reported but error actually occurs
    return learning_rate


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    learning_rate = arguments.learning_rate * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for parameter_group_ in optimizer.param_groups:
        parameter_group_['lr'] = learning_rate
    # lr cannot be fully named otherwise, no error reported but error actually occurs
    final_learning_rate = learning_rate
    return learning_rate, final_learning_rate


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_directory', default='./datasets/train_content',
                    type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_directory', default='./datasets/train_style',
                    type=str,
                    # wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str,
                    default='./experiments/vgg_normalized.pth')  # run the train.py, please download the pretrained vgg checkpoint

# training options
parser.add_argument('--save_directory', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_directory', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--learning_rate', type=float, default=5e-4)  # default 5e-4
parser.add_argument('--learning_rate_decay', type=float, default=1e-5)
parser.add_argument('--max_iteration', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--style_weight', type=float, default=10.0)  # default 10.0
parser.add_argument('--content_weight', type=float, default=7.0)  # default 7.0
parser.add_argument('--number_of_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str,
                    choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dimension', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--train_show_iteration', type=int, default=100)
arguments = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(arguments.save_directory):
    os.makedirs(arguments.save_directory)

if not os.path.exists(arguments.log_directory):
    os.mkdir(arguments.log_directory)
writer = SummaryWriter(log_dir=arguments.log_directory)  # log_dir in SummaryWriter

vgg = StyTR.vgg
vgg.load_state_dict(torch.load(arguments.vgg))
vgg = torch_neural_network.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()
Transformers = transformer.Transformer()

with torch.no_grad():
    network = StyTR.StyTrans(vgg, decoder, embedding, Transformers, arguments)
network.train()

network.to(device)
ids = torch.cuda.device_count()
network = torch_neural_network.DataParallel(network, device_ids=[
    np.array(",".join(str(i) for i in range(ids)))])

content_transform = train_transform()
style_transform = train_transform()

content_dataset = FlatFolderDataset(arguments.content_directory, content_transform)
style_dataset = FlatFolderDataset(arguments.style_directory, style_transform)

content_iteration = iter(data.DataLoader(
    content_dataset, batch_size=arguments.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=arguments.number_of_threads))
style_iteration = iter(data.DataLoader(
    style_dataset, batch_size=arguments.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=arguments.number_of_threads))

optimizer = torch.optim.Adam([
    {'params': network.module.transformer.parameters()},
    {'params': network.module.decode.parameters()},
    {'params': network.module.embedding.parameters()},
], lr=arguments.learning_rate)
# lr in torch.optim.Adam

if not os.path.exists(arguments.save_directory + "/test"):
    os.makedirs(arguments.save_directory + "/test")

for i in tqdm(range(arguments.max_iteration)):

    if i < 1e4:
        learning_rate_result, final_learning_rate = warmup_learning_rate(
            optimizer, iteration_count=i)
    else:
        learning_rate_result = adjust_learning_rate(optimizer,
                                                    iteration_count=i,
                                                    warmup_final_learning_rate=final_learning_rate)

    # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
    content_images = next(content_iteration).to(device)
    style_images = next(style_iteration).to(device)

    out, loss_content, loss_style, loss_identity1, loss_identity2 = network(
        content_images, style_images)

    if i % arguments.train_show_iteration == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
            arguments.save_directory, str(i), ".jpg"
        )
        out = torch.cat((content_images, out), 0)
        out = torch.cat((style_images, out), 0)
        save_image(out, output_name)

    loss_content = arguments.content_weight * loss_content
    loss_style = arguments.style_weight * loss_style
    loss = loss_content + loss_style + (loss_identity1 * 70) + (
                loss_identity2 * 1)

    '''
    save loss to txt file
    '''
    save_loss2txt.save_loss2txt_function(output_path_file='loss.txt', i=i,
                                         loss=loss)

    print("-total loss:", loss.sum().cpu().detach().numpy(),
          "\t-content:", loss_content.sum().cpu().detach().numpy(),
          "\t-style:", loss_style.sum().cpu().detach().numpy(),
          "\t-loss_identity1:", loss_identity1.sum().cpu().detach().numpy(),
          "\t-loss_identity2:", loss_identity2.sum().cpu().detach().numpy()
          )

    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_content.sum().item(), i + 1)
    writer.add_scalar('loss_style', loss_style.sum().item(), i + 1)
    writer.add_scalar('loss_identity1', loss_identity1.sum().item(), i + 1)
    writer.add_scalar('loss_identity2', loss_identity2.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    if (i + 1) % arguments.save_model_interval == 0 or (i + 1) == arguments.max_iteration:
        state_dictionary = network.module.transformer.state_dict()
        for key in state_dictionary.keys():
            state_dictionary[key] = state_dictionary[key].to(torch.device('cpu'))
        torch.save(state_dictionary,
                   '{:s}/transformer_iteration_{:d}.pth'.format(
                       arguments.save_directory,
                       i + 1))

        state_dictionary = network.module.decode.state_dict()
        for key in state_dictionary.keys():
            state_dictionary[key] = state_dictionary[key].to(torch.device('cpu'))
        torch.save(state_dictionary,
                   '{:s}/decoder_iteration_{:d}.pth'.format(arguments.save_directory,
                                                            i + 1))
        state_dictionary = network.module.embedding.state_dict()
        for key in state_dictionary.keys():
            state_dictionary[key] = state_dictionary[key].to(torch.device('cpu'))
        torch.save(state_dictionary,
                   '{:s}/embedding_iteration_{:d}.pth'.format(
                       arguments.save_directory,
                       i + 1))

writer.close()
