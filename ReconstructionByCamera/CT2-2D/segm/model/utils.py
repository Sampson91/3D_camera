import torch
import torch.nn as nn
import torch.nn.functional as torch_function
import math
from collections import defaultdict
import warnings
from skimage import color, io, transform
import os
from timm.models.layers import trunc_normal_
import torchvision.models as models

import segm.utils.torch as ptu
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import glob


def init_weights(map):
    if isinstance(map, nn.Linear):
        trunc_normal_(map.weight, std=0.02)
        if isinstance(map, nn.Linear) and map.bias is not None:
            nn.init.constant_(map.bias, 0)
    elif isinstance(map, nn.LayerNorm):
        nn.init.constant_(map.bias, 0)
        nn.init.constant_(map.weight, 1.0)


def resize_position_embed(position_embed, grid_old_shape, grid_new_shape,
                          number_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    position_embed_tok, position_embed_grid = (
        position_embed[:, :number_extra_tokens],
        position_embed[0, number_extra_tokens:],
    )
    if grid_old_shape is None:
        grid_shape_old_height = int(math.sqrt(len(position_embed_grid)))
        grid_shape_old_width = grid_shape_old_height
    else:
        grid_shape_old_height, grid_shape_old_width = grid_old_shape

    grid_shape_height, grid_shape_width = grid_new_shape
    position_embed_grid = position_embed_grid.reshape(1, grid_shape_old_height,
                                                      grid_shape_old_width,
                                                      -1).permute(0, 3, 1, 2)
    position_embed_grid = torch_function.interpolate(position_embed_grid, size=(
    grid_shape_height, grid_shape_width), mode="bilinear")  # ??
    position_embed_grid = position_embed_grid.permute(0, 2, 3, 1).reshape(1,
                                                                          grid_shape_height * grid_shape_width,
                                                                          -1)
    position_embed = torch.cat([position_embed_tok, position_embed_grid], dim=1)
    return position_embed


def checkpoint_filter_function(state_dictionary, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dictionary = {}
    if "model" in state_dictionary:
        # For deit models
        state_dictionary = state_dictionary["model"]
    number_extra_tokens = 1 + ("dist_token" in state_dictionary.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for key_, value_ in state_dictionary.items():
        if key_ == "pos_embed" and value_.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            value_ = resize_position_embed(
                value_,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                number_extra_tokens,
            )
        out_dictionary[key_] = value_
    return out_dictionary


def padding(image, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    height, width = image.size(2), image.size(3)
    pad_height, pad_width = 0, 0
    if height % patch_size > 0:
        pad_height = patch_size - (height % patch_size)
    if width % patch_size > 0:
        pad_width = patch_size - (width % patch_size)
    image_padded = image
    if pad_height > 0 or pad_width > 0:
        image_padded = torch_function.pad(image, (0, pad_width, 0, pad_height),
                                          value=fill_value)
    return image_padded


def unpadding(need_unpad, target_size):
    height, width = target_size
    height_pad, width_pad = need_unpad.size(2), need_unpad.size(3)
    # crop predictions on extra pixels coming from padding
    extra_height = height_pad - height
    extra_width = width_pad - width
    if extra_height > 0:
        need_unpad = need_unpad[:, :, :-extra_height]
    if extra_width > 0:
        need_unpad = need_unpad[:, :, :, :-extra_width]
    return need_unpad


def resize(image, smaller_size):
    height, width = image.shape[2:]
    if height < width:
        ratio = width / height
        height_resize, width_resize = smaller_size, ratio * smaller_size
    else:
        ratio = height / width
        height_resize, width_resize = ratio * smaller_size, smaller_size
    if min(height, width) < smaller_size:
        image_resize = torch_function.interpolate(image, (
        int(height_resize), int(width_resize)), mode="bilinear")
    else:
        image_resize = image
    return image_resize


def sliding_window(image, flip, window_size, window_stride):
    batch, channal, height, width = image.shape
    window_size = window_size

    windows = {"crop": [], "anchors": []}
    height_anchors = torch.arange(0, height, window_stride)
    width_anchors = torch.arange(0, width, window_stride)
    height_anchors = [h.item() for h in height_anchors if
                      h < height - window_size] + [height - window_size]
    width_anchors = [w.item() for w in width_anchors if
                     w < width - window_size] + [width - window_size]
    for height_anchor_ in height_anchors:
        for width_anchor_ in width_anchors:
            window = image[:, :, height_anchor_: height_anchor_ + window_size,
                     width_anchor_: width_anchor_ + window_size]
            windows["crop"].append(window)
            windows["anchors"].append((height_anchor_, width_anchor_))
    windows["flip"] = flip
    windows["shape"] = (height, width)
    return windows


def merge_windows(windows, window_size, ori_shape):
    window_size = window_size
    image_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    channal = image_windows[0].shape[0]
    height, width = windows["shape"]
    flip = windows["flip"]

    logit = torch.zeros((channal, height, width), device=image_windows.device)
    count = torch.zeros((1, height, width), device=image_windows.device)
    for window, (height_anchors, width_anchors) in zip(image_windows, anchors):
        logit[:, height_anchors: height_anchors + window_size,
        width_anchors: width_anchors + window_size] += window
        count[:, height_anchors: height_anchors + window_size,
        width_anchors: width_anchors + window_size] += 1
    logit = logit / count
    logit = torch_function.interpolate(
        logit.unsqueeze(0),
        ori_shape,
        mode="bilinear",
    )[0]
    if flip:
        logit = torch.flip(logit, (2,))
    result = torch_function.softmax(logit, 0)
    return result


def inference(
        model,
        images,
        images_metas,
        original_shape,
        window_size,
        window_stride,
        batch_size,
):
    colors = model.number_of_colors
    seg_map = torch.zeros((colors, original_shape[0], original_shape[1]),
                          device=ptu.device)
    for image, image_metas in zip(images, images_metas):
        image = image.to(ptu.device)
        image = resize(image, window_size)
        flip = image_metas["flip"]
        windows = sliding_window(image, flip, window_size, window_stride)
        crops = torch.stack(windows.pop("crop"))[:, 0]
        batch = len(crops)
        batch_step = batch_size
        seg_maps = torch.zeros((batch, colors, window_size, window_size),
                               device=image.device)
        with torch.no_grad():
            for i in range(0, batch, batch_step):
                seg_maps[i: i + batch_step] = model.forward(crops[i: i + batch_step])
        windows["seg_maps"] = seg_maps
        image_seg_map = merge_windows(windows, window_size, original_shape)
        seg_map += image_seg_map
    seg_map /= len(images)
    return seg_map


def number_of_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    number_of_parameters = sum(
        [torch.prod(torch.tensor(parameter_.size())) for parameter_ in model_parameters])
    return number_of_parameters.item()


#########  encode AB ###############
class SoftEncodeAB:
    def __init__(self, cielab, neighbours=5, sigma=5.0, device='cuda'):
        self.cielab = cielab
        self.q_to_ab = torch.from_numpy(self.cielab.q_to_ab).to(device)

        self.neighbours = neighbours
        self.sigma = sigma

    def __call__(self, ab):
        numbers, _, height, width = ab.shape

        map = numbers * height * width

        # find nearest neighbours
        ab_ = ab.permute(1, 0, 2, 3).reshape(2, -1)
        query_to_ab = self.q_to_ab.type(ab_.dtype)

        cdist = torch.cdist(query_to_ab, ab_.t())

        neural_networks = cdist.argsort(dim=0)[:self.neighbours, :]

        # gaussian weighting
        neural_network_gauss = ab.new_zeros(self.neighbours, map)

        for i in range(self.neighbours):
            neural_network_gauss[i, :] = self._gauss_eval(
                query_to_ab[neural_networks[i, :], :].t(), ab_, self.sigma)

        neural_network_gauss /= neural_network_gauss.sum(dim=0, keepdim=True)

        # expand
        bins = self.cielab.gamut.EXPECTED_SIZE

        query = ab.new_zeros(bins, map)

        query[neural_networks, torch.arange(map).repeat(self.neighbours, 1)] = neural_network_gauss
        # return: [bs, 313, 256, 256]
        return query.reshape(bins, numbers, height, width).permute(1, 0, 2, 3)

    @staticmethod
    def _gauss_eval(need_gauss_evaluation, mu, sigma):
        normalization = 1 / (2 * math.pi * sigma)

        return normalization * torch.exp(
            -torch.sum((need_gauss_evaluation - mu) ** 2, dim=0) / (2 * sigma ** 2))


########### CIELAB #####################################
# _SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
_SOURCE_DIRECTORY = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))  # last dir
_RESOURCE_DIRECTORY = os.path.join(_SOURCE_DIRECTORY, 'resources')


def lab_to_rgb(image):
    assert image.dtype == np.float32

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        return (255 * np.clip(color.lab2rgb(image), 0, 1)).astype(np.uint8)


def get_resource_path(path):
    return os.path.join(_RESOURCE_DIRECTORY, path)


class ABGamut:
    RESOURCE_POINTS = get_resource_path('ab-gamut.npy')
    RESOURCE_PRIOR = get_resource_path('q-prior.npy')

    DTYPE = np.float32
    EXPECTED_SIZE = 313

    def __init__(self):
        self.points = np.load(self.RESOURCE_POINTS, allow_pickle=True).astype(
            self.DTYPE)
        self.prior = np.load(self.RESOURCE_PRIOR, allow_pickle=True).astype(
            self.DTYPE)

        assert self.points.shape == (self.EXPECTED_SIZE, 2)
        assert self.prior.shape == (self.EXPECTED_SIZE,)


class CIELAB:
    Luminance_MEAN = 50

    AB_BINSIZE = 10
    AB_RANGE = [-110 - AB_BINSIZE // 2, 110 + AB_BINSIZE // 2, AB_BINSIZE]
    AB_DTYPE = np.float32

    Query_DTYPE = np.int64

    RGB_RESOLUTION = 101
    RGB_RANGE = [0, 1, RGB_RESOLUTION]
    RGB_DTYPE = np.float64

    def __init__(self, gamut=None):
        self.gamut = gamut if gamut is not None else ABGamut()

        a, b, self.ab = self._get_ab()

        self.ab_gamut_mask = self._get_ab_gamut_mask(
            a, b, self.ab, self.gamut)

        self.ab_to_q = self._get_ab_to_q(self.ab_gamut_mask)
        self.q_to_ab = self._get_q_to_ab(self.ab, self.ab_gamut_mask)

    @classmethod
    def _get_ab(cls):
        a = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        b = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)

        b_, a_ = np.meshgrid(a, b)
        ab = np.dstack((a_, b_))

        return a, b, ab

    @classmethod
    def _get_ab_gamut_mask(cls, a, b, ab, gamut):
        ab_gamut_mask = np.full(ab.shape[:-1], False, dtype=bool)

        a = np.digitize(gamut.points[:, 0], a) - 1
        b = np.digitize(gamut.points[:, 1], b) - 1

        for a_, b_ in zip(a, b):
            ab_gamut_mask[a_, b_] = True

        return ab_gamut_mask

    @classmethod
    def _get_ab_to_q(cls, ab_gamut_mask):
        ab_to_q = np.full(ab_gamut_mask.shape, -1, dtype=cls.Query_DTYPE)

        ab_to_q[ab_gamut_mask] = np.arange(np.count_nonzero(ab_gamut_mask))

        return ab_to_q

    @classmethod
    def _get_q_to_ab(cls, ab, ab_gamut_mask):
        return ab[ab_gamut_mask] + cls.AB_BINSIZE / 2

    @classmethod
    def _plot_ab_matrix(cls, mat, pixel_borders=False, axial=None, title=None):
        if axial is None:
            _, axial = plt.subplots()

        image_show = partial(axial.imshow,
                             np.flip(mat, axis=0),
                             extent=[*cls.AB_RANGE[:2]] * 2)

        if len(mat.shape) < 3 or mat.shape[2] == 1:
            image = image_show(cmap='jet')

            figure = plt.gcf()
            figure.colorbar(image, cax=figure.add_axes())
        else:
            image_show()

        # set title
        if title is not None:
            axial.set_title(title)

        # set axes labels
        axial.set_xlabel("$b$")
        axial.set_ylabel("$a$")

        # minor ticks
        tick_min_minor = cls.AB_RANGE[0]
        tick_max_minor = cls.AB_RANGE[1]

        if pixel_borders:
            axial.set_xticks(
                np.linspace(tick_min_minor, tick_max_minor, mat.shape[1] + 1),
                minor=True)

            axial.set_yticks(
                np.linspace(tick_min_minor, tick_max_minor, mat.shape[0] + 1),
                minor=True)

            axial.grid(which='minor',
                       color='w',
                       linestyle='-',
                       linewidth=2)

        # major ticks
        tick_min_major = tick_min_minor + cls.AB_BINSIZE // 2
        tick_max_major = tick_max_minor - cls.AB_BINSIZE // 2

        axial.set_xticks(np.linspace(tick_min_major, tick_max_major, 5))
        axial.set_yticks(np.linspace(tick_min_major, tick_max_major, 5))

        # some of this will be obscured by the minor ticks due to a five year
        # old matplotlib bug...
        axial.grid(which='major',
                   color='k',
                   linestyle=':',
                   dashes=(1, 4))

        # tick marks
        for axial_ in axial.xaxis, axial.yaxis:
            axial_.set_ticks_position('both')

        axial.tick_params(axis='both', which='major', direction='in')
        axial.tick_params(axis='both', which='minor', length=0)

        # limits
        limit_min = tick_min_major - cls.AB_BINSIZE
        limit_max = tick_max_major + cls.AB_BINSIZE

        axial.set_xlim([limit_min, limit_max])
        axial.set_ylim([limit_min, limit_max])

        # invert y-axis
        axial.invert_yaxis()

    def bin_ab(self, ab):
        ab_discrete = ((ab + 110) / self.AB_RANGE[2]).astype(int)

        a, b = np.hsplit(ab_discrete.reshape(-1, 2), 2)
        q = self.ab_to_q[a, b].reshape(*ab.shape[:2])

        return q

    def plot_ab_gamut(self, luminance=50, axial=None):
        assert luminance >= 50 and luminance <= 100

        # construct Lab color space slice for given L
        luminance = np.full(self.ab.shape[:2], luminance, dtype=self.ab.dtype)
        color_space_lab = np.dstack((luminance, self.ab))

        # convert to RGB
        color_space_rgb = lab_to_rgb(color_space_lab)

        # mask out of gamut colors
        color_space_rgb[~self.ab_gamut_mask, :] = 255

        # display color space
        self._plot_ab_matrix(color_space_rgb,
                             pixel_borders=True,
                             axial=axial,
                             title=r"$RGB(a, b \mid L = {})$".format(luminance))

    def plot_empirical_distribution(self, dataset, axial=None, verbose=False):
        # accumulate ab values
        ab_accuracy = np.zeros([self.AB_RANGE[1] - self.AB_RANGE[0]] * 2)

        for i in range(len(dataset)):
            image = dataset[i]

            if verbose:
                formation = "\rprocessing image {}/{}"

                print(formation.format(i + 1, len(dataset)),
                      end=('\n' if i == len(dataset) - 1 else ''),
                      flush=True)

            image = np.moveaxis(image, 0, -1)
            ab_rounded = np.round(image[:, :, 1:].reshape(-1, 2)).astype(int)
            ab_offset = ab_rounded - self.AB_RANGE[0]

            np.add.at(ab_accuracy, tuple(np.split(ab_offset, 2, axis=1)), 1)

        # convert to log scale
        ab_accuracy[ab_accuracy == 0] = np.nan

        ab_acc_log = np.log10(ab_accuracy) - np.log10(len(dataset))

        # display distribution
        self._plot_ab_matrix(ab_acc_log, axial=axial, title=r"$log(P(a, b))$")


class AnnealedMeanDecodeQuery:
    def __init__(self, cielab, special_T, device='cuda'):
        self.q_to_ab = torch.from_numpy(cielab.q_to_ab).to(device)

        self.special_T = special_T

    def __call__(self, query, is_actual=False, applied=False):
        if self.special_T == 0:
            # makeing this a special case is somewhat ugly but I have found
            # no way to make this a special case of the branch below (in
            # NumPy that would be trivial)
            ab = self._unbin(self._mode(query))
        else:
            if is_actual is False:
                query = self._annealed_softmax(query, applied=applied)

            a = self._annealed_mean(query, 0)
            b = self._annealed_mean(query, 1)
            ab = torch.cat((a, b), dim=1)

        return ab.type(query.dtype)

    def _mode(self, query):
        return query.maximum(dim=1, keepdim=True)[1]

    def _unbin(self, query):
        _, _, height, width = query.shape  # [bs, 1, h, w]

        ab = torch.stack([
            self.q_to_ab.index_select(
                0, query_.flatten()
            ).reshape(height, width, 2).permute(2, 0, 1)

            for query_ in query
        ])

        return ab

    def _annealed_softmax(self, query, applied=False, change_mask=None):
        query_exp = torch.exp(query / self.special_T)
        if not applied:
            query_softmax = query_exp / query_exp.sum(dim=1, keepdim=True)
        else:
            query_softmax = query_exp / query_exp.sum(dim=1,
                                          keepdim=True)  # [bs, 313, 256, 256]

        return query_softmax

    def _annealed_mean(self, query, dimension):
        annealed_mean = torch.tensordot(query, self.q_to_ab[:, dimension], dims=((1,), (0,)))

        return annealed_mean.unsqueeze(dim=1)


########################### VGG ######################################################
def load_model(model_name, model_directory):
    assert os.path.exists(model_directory)
    model = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(model_directory, '%s-[a-z0-9]*.pth' % model_name)
    model_path = glob.glob(path_format)[0]
    model.load_state_dict(torch.load(model_path))
    return model


class VGG19(torch.nn.Module):
    def __init__(self, requires_gradient=False):
        super().__init__()
        model_dir = os.path.join(os.getcwd(), 'segm/resources')
        # model_dir = os.path.join(optimizer.pretrained_dir, 'vgg')
        model = load_model('vgg19', model_dir)
        vgg_pretrained_features = model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for Vgg_neural_network in range(2):
            self.slice1.add_module(str(Vgg_neural_network), vgg_pretrained_features[Vgg_neural_network])
        for Vgg_neural_network in range(2, 7):
            self.slice2.add_module(str(Vgg_neural_network), vgg_pretrained_features[Vgg_neural_network])
        for Vgg_neural_network in range(7, 12):
            self.slice3.add_module(str(Vgg_neural_network), vgg_pretrained_features[Vgg_neural_network])
        for Vgg_neural_network in range(12, 21):
            self.slice4.add_module(str(Vgg_neural_network), vgg_pretrained_features[Vgg_neural_network])

        for Vgg_neural_network in range(21, 30):
            self.slice5.add_module(str(Vgg_neural_network), vgg_pretrained_features[Vgg_neural_network])
        if not requires_gradient:
            for parameter_ in self.parameters():
                parameter_.requires_grad = False

    def forward(self, VGG_neural_network):
        relu1 = self.slice1(VGG_neural_network)
        relu2 = self.slice2(relu1)
        relu3 = self.slice3(relu2)
        relu4 = self.slice4(relu3)
        relu5 = self.slice5(relu4)
        out = [relu1, relu2, relu3, relu4, relu5]
        return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    z_int = torch.max(torch.Tensor((0,)).to(lab.device), z_int)
    out = torch.cat(
        (x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]),
        dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    mask = mask.to(lab.device)
    out = (out ** 3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)
    score = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    score = score.to(out.device)
    out = out * score
    return out


def xyz2rgb(xyz):
    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :,
                                                    :] - 0.49853633 * xyz[:, 2,
                                                                      :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :,
                                                  :] + .04155593 * xyz[:, 2, :,
                                                                   :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :,
                                                  :] + 1.05731107 * xyz[:, 2, :,
                                                                    :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]),
                    dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))
    mask = (rgb > .0031308).type(torch.FloatTensor)
    mask = mask.to(xyz.device)
    rgb = (1.055 * (rgb ** (1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (
                1 - mask)
    return rgb


def lab2rgb(image_lab):
    # img_lab: torch.tensor().
    out = xyz2rgb(lab2xyz(image_lab))
    return out
