import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics, classify_accuracy
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu
import torch.nn as nn
from torch.nn.functional import log_softmax
import torch.nn.functional as torch_functional
import numpy as np
from torchvision.models.inception import inception_v3
from torchvision.transforms import transforms
import os
from PIL import Image
from skimage import color, io
import warnings
from torch.autograd import Variable
from segm.metrics import InceptionV3FrechetInceptionDistance, INCEPTION_V3, \
    get_activations, calculate_frechet_distance, calculate_activation_statistics
import random
import cv2


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels, add_mask=None):
        batch, number_of_colors, height, width = outputs.shape
        reshape_out = outputs.permute(0, 2, 3, 1).contiguous().view(
            batch * height * width, number_of_colors)
        reshape_label = labels.permute(0, 2, 3, 1).contiguous().view(
            batch * height * width, number_of_colors)  # [-1, 313]
        after_softmax = torch_functional.softmax(reshape_out, dim=1)
        # mask = add_mask.view(-1, number_of_colors)
        mask = after_softmax.clone()
        after_softmax = after_softmax.masked_fill(mask == 0, 1)
        out_softmax = torch.log(after_softmax)

        normalization = reshape_label.clone()
        normalization = normalization.masked_fill(reshape_label == 0, 1)
        log_normalization = torch.log(normalization)

        loss = -torch.sum((out_softmax - log_normalization) * reshape_label) / (
                batch * height * width)
        return loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = utils.VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, source, target):
        source_vgg, target_vgg = self.vgg(source), self.vgg(target)
        loss = 0
        for i in range(len(source_vgg)):
            loss += self.weights[i] * self.criterion(source_vgg[i],
                                                     target_vgg[i].detach())
        return loss


def functional_conv2d(image, input_channel=2, output_channel=2):
    convolution_option = nn.Conv2d(input_channel, output_channel, 3, padding=1,
                                   bias=False).to(image.device)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                            dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape(
        (1, 1, 3, 3))  # suitable for ab two channels.
    sobel_kernel_tensor = torch.from_numpy(sobel_kernel).repeat(output_channel,
                                                                input_channel,
                                                                1, 1)
    convolution_option.weight.data = sobel_kernel_tensor.to(image.device)
    edge_detect = convolution_option(Variable(image))
    return edge_detect


def train_one_epoch(
        model,
        data_loader,
        optimizer,
        learning_rate_scheduler,
        epoch,
        amp_autocast,
        loss_scaler,
        add_mask,
        add_l1_loss,
        l1_weight,
        partial_finetune,
        l1_convolution,
        l1_linear,
        add_edge,
        edge_loss_weight,
        without_classification,
        log_directory,
        preview_directory,
        dataset_directory
):
    if not without_classification:
        criterion = CrossEntropyLoss2d()
    if add_l1_loss:
        loss_function_l1 = nn.SmoothL1Loss()
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_frequency = 50

    model.train()
    if partial_finetune:
        if hasattr(model, "module"):
            for block_layer in range(6):
                model.module.encoder.blocks[block_layer].eval()
        else:
            for block_layer in range(6):
                model.encoder.blocks[block_layer].eval()

    data_loader.set_epoch(epoch)
    number_of_updates = epoch * len(data_loader)

    train_loss_total, train_l1_total = 0, 0

    random_batch = None
    batch = None
    rand_ab_prediction = None

    for batch in logger.log_every(data_loader, print_frequency, header):
        image_luminance, image_ab, key, image_mask = batch

        image_luminance = image_luminance.to(ptu.device)
        image_ab = image_ab.to(ptu.device)
        if add_mask: image_mask = image_mask.to(ptu.device)

        with amp_autocast():
            if add_mask:
                ab_prediction, query_prediction, query_actual, out_feature = model.forward(
                    image_luminance, image_ab,
                    image_mask)  # out_feature: [B, 2, 256, 256]
            else:
                ab_prediction, query_prediction, query_actual, out_feature = model.forward(
                    image_luminance, image_ab, None)

            if not without_classification:  # default False.
                loss = criterion(query_prediction, query_actual)
            else:
                loss = 0

            if add_l1_loss:
                if l1_convolution:
                    normalization_ab = image_ab / 110.  # [-1, 1]
                    loss_l1 = loss_function_l1(normalization_ab, out_feature)
                elif l1_linear:
                    normalization_ab = image_ab / 110.
                    loss_l1 = loss_function_l1(normalization_ab, out_feature)
                loss += loss_l1 * l1_weight

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()
        loss_value = loss.item()
        train_loss_total += loss_value
        if add_l1_loss:
            train_l1_total += (loss_l1 * l1_weight).item()
        else:
            train_l1_total = 0
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value),
                  force=True)

        number_of_updates += 1
        # if num_updates % 5000 == 0 and ptu.dist_rank == 0:
        #     model_without_ddp = model
        #     if hasattr(model, "module"):
        #         model_without_ddp = model.module
        #     snapshot = dict(
        #         model=model_without_ddp.state_dict(),
        #         optimizer=optimizer.state_dict(),
        #         number_of_colors=model_without_ddp.number_of_colors,
        #         learning_rate_scheduler=learning_rate_scheduler.state_dict(),
        #     )
        #     if loss_scaler is not None:
        #         snapshot["loss_scaler"] = loss_scaler.state_dict()
        #     snapshot["epoch"] = epoch
        #     save_path = os.path.join(log_directory, 'checkpoint_epoch_%d_iter_%d.pth' % (epoch, num_updates))
        #     torch.save(snapshot, save_path)
        #     print('save model into:', save_path)

        learning_rate_scheduler.step_update(num_updates=number_of_updates)

        # randomly keep a batch
        if random.randrange(0, 100, 1) < 50:
            random_batch = batch
            rand_ab_prediction = ab_prediction

        torch.cuda.synchronize()

    if not random_batch:
        random_batch = batch
        rand_ab_prediction = ab_prediction

    '''
    save preview images during training
    '''
    with torch.no_grad():
        image_luminance, image_ab, key, image_mask = random_batch
        filename = key[1]
        image_luminance = image_luminance.to(ptu.device)
        image_ab = image_ab.to(ptu.device)

        save_preview_images(image_luminance, image_ab, rand_ab_prediction,
                            filename, preview_directory, epoch,
                            dataset_directory)

    logger.update(
        loss=train_loss_total / len(data_loader),
        loss_l1=train_l1_total / len(data_loader),
        learning_rate=optimizer.param_groups[0]["lr"],  # lr optimizer 内部
    )
    return logger


def save_preview_images(image_luminance, image_ab, ab_prediction, filenames,
                        directory, epoch, real_image_path):
    image_lab = torch.cat((image_luminance, ab_prediction.detach()),
                          dim=1).cpu()
    batch_size = image_lab.size(0)
    fake_rgb_list, real_rgb_list, only_rgb_list = [], [], []

    image_lab_numpy = image_lab[0].numpy().transpose(1, 2, 0)  # np.float32
    image_rgb = lab_to_rgb(image_lab_numpy)  # np.uint8      # [0-255]
    fake_rgb_list.append(image_rgb)

    filename = str(epoch) + '_' + filenames[0]

    image_path = os.path.join(directory, filename)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for root_, directory_, file_ in os.walk(real_image_path):
            if filenames[0] in file_:
                real_image_path_file = os.path.join(root_, filenames[0])
                real_image = Image.open(real_image_path_file).convert('RGB')
                real_image_gray = Image.open(real_image_path_file).convert('L')
                real_image_gray = real_image_gray.convert('RGB')
                height_fake = fake_rgb_list[0].shape[0]
                width_fake = fake_rgb_list[0].shape[1]

                # resize the size of real image to fake
                real_image = real_image.resize((width_fake, height_fake))
                real_image_gray = real_image_gray.resize((width_fake, height_fake))

                real_image = np.array(real_image)
                real_image_gray = np.array(real_image_gray)

                preview_image = np.hstack((real_image, real_image_gray, fake_rgb_list[0]))
                io.imsave(image_path, preview_image.astype(np.uint8))
                break

    full_directory = os.path.join(os.getcwd(), directory)
    full_directory_with_file = os.path.join(full_directory, filename)
    print('image saved:', full_directory_with_file)


@torch.no_grad()
def evaluate(
        epoch,
        model,
        data_loader,
        window_size,
        window_stride,
        amp_autocast,
        add_mask,
        add_l1_loss,
        l1_weight,
        l1_convolution,
        l1_linear,
        add_feature_match,
        feature_match_weight,
        log_directory=None,
        diversity_index=0,
        save_directory=None
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_frequency = 50
    if log_directory is not None:
        # save_directory = os.path.join(log_directory, 'color_token_nums_78')
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
    model.eval()
    total_peak_signal_to_noise_ratio_classification, total_peak_signal_to_noise_ratio_regression, frechet_inception_distance_score = 0, 0, 0
    with torch.no_grad():
        for batch in logger.log_every(data_loader, print_frequency, header):
            image_luminamce, image_ab, filename, image_mask = batch
            image_luminamce = image_luminamce.to(ptu.device)
            image_ab = image_ab.to(ptu.device)
            if add_mask: image_mask = image_mask.to(ptu.device)

            with amp_autocast():
                if add_mask:
                    ab_prediction, query_prediction, query_actual, out_feature = model_without_ddp.inference(
                        image_luminamce, image_ab, image_mask)
                else:
                    ab_prediction, query_prediction, query_actual, out_feature = model_without_ddp.inference(
                        image_luminamce, image_ab, None)

            if log_directory is not None:
                if ab_prediction is not None:
                    save_images(image_luminamce, image_ab, ab_prediction,
                                filename, save_directory)

    logger.update(
        eval_psnr_cls=total_peak_signal_to_noise_ratio_classification / len(
            data_loader),
        eval_psnr_reg=total_peak_signal_to_noise_ratio_regression / len(
            data_loader),
        eval_fid=frechet_inception_distance_score)
    return logger


def lab_to_rgb(image):
    assert image.dtype == np.float32
    return (255 * np.clip(color.lab2rgb(image), 0, 1)).astype(np.uint8)


def save_images(image_luminance, image_ab, ab_prediction, filenames, directory):
    image_lab = torch.cat((image_luminance, ab_prediction.detach()),
                          dim=1).cpu()
    batch_size = image_lab.size(0)
    fake_rgb_list, real_rgb_list, only_rgb_list = [], [], []
    for j in range(batch_size):
        image_lab_numpy = image_lab[j].numpy().transpose(1, 2, 0)  # np.float32
        image_rgb = lab_to_rgb(image_lab_numpy)  # np.uint8      # [0-255]
        fake_rgb_list.append(image_rgb)

        image_path = os.path.join(directory, 'fake_' + filenames[j])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            io.imsave(image_path, fake_rgb_list[j].astype(np.uint8))

        full_directory = os.path.join(os.getcwd(), directory)
        full_directory_with_file = os.path.join(full_directory, filenames[j])
        print('image saved:', full_directory_with_file)


def calculate_peak_signal_to_noise_ratio(image_luminance, image_ab,
                                         ab_prediction, out_feature):
    real_lab = torch.cat((image_luminance, image_ab), dim=1).cpu()
    if ab_prediction is not None:
        fake_lab = torch.cat((image_luminance, ab_prediction.detach()),
                             dim=1).cpu()
    if out_feature is not None:
        fake_lab_regression = torch.cat(
            (image_luminance, (out_feature * 110).detach()), dim=1).cpu()
    base_size = real_lab.size(0)
    assert base_size == 1
    ##############
    peak_signal_to_noise_ratio_classification, peak_signal_to_noise_ratio_regression = 0, 0
    for j in range(base_size):
        real_lab_numpy = real_lab[j].numpy().transpose(1, 2, 0)
        real_rgb = lab_to_rgb(real_lab_numpy)
        if ab_prediction is not None:
            fake_lab_numpy = fake_lab[j].numpy().transpose(1, 2, 0)
            fake_peak_signal_to_noise_ratio = lab_to_rgb(fake_lab_numpy)
            each_peak_signal_to_noise_ratio = calculate_peak_signal_to_noise_ratio_numpy(
                fake_peak_signal_to_noise_ratio, real_rgb)
            peak_signal_to_noise_ratio_classification += each_peak_signal_to_noise_ratio

        if out_feature is not None:
            fake_lab_regression = fake_lab_regression[j].numpy().transpose(1, 2,
                                                                           0)
            fake_rgb_regression = lab_to_rgb(fake_lab_regression)
            each_peak_signal_to_noise_ratio_regression = calculate_peak_signal_to_noise_ratio_numpy(
                fake_rgb_regression, real_rgb)
            peak_signal_to_noise_ratio_regression += each_peak_signal_to_noise_ratio_regression
    peak_signal_to_noise_ratio_classification = peak_signal_to_noise_ratio_classification / base_size
    peak_signal_to_noise_ratio_regression = peak_signal_to_noise_ratio_regression / base_size

    return peak_signal_to_noise_ratio_classification, peak_signal_to_noise_ratio_regression


def calculate_peak_signal_to_noise_ratio_numpy(image1, image2):
    import numpy as np
    squared_error_map = (1. * image1 - image2) ** 2
    current_mean_squared_error = np.mean(squared_error_map)
    return 20 * np.log10(255. / np.sqrt(current_mean_squared_error))


def lab2xyz(lab):
    # xyz is the coordinate in lab
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    if (z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat(
        (x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]),
        dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if (out.is_cuda):
        mask = mask.cuda()

    out = (out ** 3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)

    score = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    score = score.to(out.device)

    out = out * score
    return out


def xyz2rgb(xyz):
    # xyz is the coordinate in lab
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
    rgb = torch.max(rgb, torch.zeros_like(
        rgb))  # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if (rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055 * (rgb ** (1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (
            1 - mask)

    return rgb
