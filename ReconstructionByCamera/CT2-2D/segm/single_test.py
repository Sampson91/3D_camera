import os
from PIL import Image
import numpy as np
from skimage import color
import torch
import pickle
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
import torch.nn as nn
import torch.distributed
import warnings
import io

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config

from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import number_of_parameters

from timm.utils import NativeScaler
from contextlib import suppress
import os

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate
import collections
import skimage


# from segm.engine import save_imgs


def rgb_to_lab(image):
    assert image.dtype == np.uint8
    return color.rgb2lab(image).astype(np.float32)


def numpy_to_torch(image):
    tensor = torch.from_numpy(np.moveaxis(image, -1, 0))  # [c, h, w]
    return tensor.type(torch.float32)


def load_mask(mask_luminance_range_number):
    file_open = open('/userhome/SUN_text2img/ImageNet/mask_prior.pickle', 'rb')
    load_dictionary = pickle.load(file_open)
    mask_luminance = np.zeros((mask_luminance_range_number, 313)).astype(
        np.bool)
    for key in range(101):
        for range_number_ in range(mask_luminance_range_number):
            start_key = range_number_ * (100 // mask_luminance_range_number)
            end_key = (range_number_ + 1) * (100 // mask_luminance_range_number)
            if start_key <= key < end_key:
                mask_luminance[range_number_, :] += load_dictionary[key].astype(np.bool)
                break
    mask_luminance = mask_luminance.astype(np.float32)
    return mask_luminance


@click.command(help="")
@click.option("--log-directory", type=str, help="logging directory")
@click.option("--dataset", default='coco', type=str)
@click.option('--dataset_directory', default='/userhome/sjm/ImageNet', type=str)
@click.option("--image-size", default=256, type=int, help="dataset resize size")
@click.option("--crop-size", default=256, type=int)
@click.option("--window-size", default=256, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="vit_tiny_patch16_384",
              type=str)  # try this, and freeze first several blocks.
@click.option("--decoder", default="mask_transformer", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-learning_rate", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--evaluation-frequency", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
@click.option('--local_rank', type=int)
@click.option('--only_test', type=bool, default=True)
@click.option('--add_mask', type=bool, default=True)  # valid
@click.option('--partial_finetune', type=bool,
              default=False)  # compare validation, last finetune all blocks.
@click.option('--add_l1_loss', type=bool,
              default=True)  # add after classification.
@click.option('--l1_weight', type=float, default=10)
@click.option('--color_position', type=bool,
              default=False)  # add color position in color token.
@click.option('--change_mask', type=bool,
              default=False)  # change mask, omit the attention between color tokens.
@click.option('--color_as_condition', type=bool,
              default=False)  # use self-attn to embedding color tokens, and use color to represent patch tokens.
@click.option('--multi_scaled', type=bool,
              default=False)  # multi-scaled decoder.
@click.option('--downchannel', type=bool,
              default=False)  # multi-scaled, upsample+downchannel. (should be correct??)
@click.option('--add_convolution', type=bool,
              default=True)  # add conv after transformer blocks.
@click.option('--before_classify', type=bool,
              default=False)  # classification at 16x16 resolution, and use CNN upsampler for 256x256, then use l1-loss.
@click.option('--l1_convolution', type=bool,
              default=True)  # patch--upsample--> [B, 256x256, 180]--conv3x3-> [B, 256x256, 2]
@click.option('--l1_linear', type=bool,
              default=False)  # pacth: [B, 16x16, 180]---linear-> [B, 16x16, 2x16x16]
@click.option('--add_feature_match', type=bool,
              default=False)  # add Feature matching loss.
@click.option('--feature_match_weight', type=float, default=1)
@click.option('--add_edge', type=bool,
              default=False)  # add sobel-conv to extract edge.
@click.option('--edge_loss_weight', type=float,
              default=0.05)  # edge_loss_weight.
@click.option('--mask_luminance_range_number', type=int,
              default=4)  # mask for L ranges: 4, 10, 20, 50, 100
@click.option('--number_of_blocks', type=int,
              default=1)  # per block have 2 layers. block_num = 2
@click.option('--without_colorattn', type=bool, default=False)
@click.option('--without_colorquery', type=bool, default=False)
@click.option('--without_classification', type=bool, default=False)
def test_func(
        log_directory,
        dataset,
        dataset_directory,
        image_size,
        crop_size,
        window_size,
        window_stride,
        backbone,
        decoder,
        optimizer,
        scheduler,
        weight_decay,
        dropout,
        drop_path,
        batch_size,
        epochs,
        learning_rate,
        normalization,
        evaluation_frequency,
        amp,
        resume,
        local_rank,
        only_test,
        add_mask,
        partial_finetune,
        add_l1_loss,
        l1_weight,
        color_position,
        change_mask,
        color_as_condition,
        multi_scaled,
        downchannel,
        add_convolution,
        before_classify,
        l1_convolution,
        l1_linear,
        add_feature_match,
        feature_match_weight,
        add_edge,
        edge_loss_weight,
        mask_luminance_range_number,
        number_of_blocks,
        without_colorattn,
        without_colorquery,
        without_classification,
):
    # start distributed mode
    ptu.set_gpu_mode(True, local_rank)
    # distributed.init_process()
    torch.distributed.init_process_group(backend="nccl")

    # set up configuration
    loaded_config = config.load_config()
    model_config = loaded_config["model"][backbone]
    dataset_config = loaded_config["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_config = loaded_config["decoder"]["mask_transformer"]
    else:
        decoder_config = loaded_config["decoder"][decoder]

    # model config
    if not image_size:
        image_size = dataset_config["image_size"]  # 256
    if not crop_size:
        crop_size = dataset_config.get("crop_size", image_size)  # 256
    if not window_size:
        window_size = dataset_config.get("window_size", image_size)
    if not window_stride:
        window_stride = dataset_config.get("window_stride", image_size)
    if not dataset_directory:
        dataset_directory = dataset_config.get('dataset_directory', None)

    model_config["image_size"] = (crop_size, crop_size)
    model_config["backbone"] = backbone
    model_config["dropout"] = dropout
    model_config["drop_path_rate"] = drop_path
    decoder_config["name"] = decoder
    model_config["decoder"] = decoder_config

    # dataset config
    world_batch_size = dataset_config["batch_size"]
    number_of_epochs = dataset_config["epochs"]
    learning_rate = dataset_config["learning_rate"]

    if batch_size:
        world_batch_size = batch_size
    if epochs:
        number_of_epochs = epochs
    if learning_rate:
        learning_rate = learning_rate
    if evaluation_frequency is None:
        evaluation_frequency = dataset_config.get("evaluation_frequency", 1)

    if normalization:
        model_config["normalization"] = normalization

    # experiment config
    batch_size = world_batch_size // ptu.world_size
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=image_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_config["normalization"],
            split="train",
            number_of_workers=10,
            dataset_directory=dataset_directory,
            add_mask=add_mask,
            patch_size=model_config["patch_size"],
            change_mask=change_mask,
            multi_scaled=multi_scaled,
            mask_number=mask_luminance_range_number,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            number_of_epochs=number_of_epochs,
            evaluation_frequency=evaluation_frequency,
        ),
        optimizer_kwargs=dict(
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
            # momentum=0.9,
            clip_gradient=None,
            sched=scheduler,
            epochs=number_of_epochs,
            min_learning_rate=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_config,
        amp=amp,
        log_directory=log_directory,
        inference_kwargs=dict(
            image_size=image_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_directory / 'checkpoint_epoch_0_psnrcls_22.8164_psnrreg_24.5049.pth'

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["number_of_colors"] = 313

    net_kwargs['partial_finetune'] = partial_finetune
    net_kwargs['decoder']['add_l1_loss'] = add_l1_loss
    net_kwargs['decoder']['color_position'] = color_position
    net_kwargs['decoder']['change_mask'] = change_mask
    net_kwargs['decoder']['color_as_condition'] = color_as_condition
    net_kwargs['decoder']['multi_scaled'] = multi_scaled
    net_kwargs['decoder']['crop_size'] = crop_size
    net_kwargs['decoder']['downchannel'] = downchannel
    net_kwargs['decoder']['add_convolution'] = add_convolution
    net_kwargs['decoder']['before_classify'] = before_classify
    net_kwargs['decoder']['l1_convolution'] = l1_convolution
    net_kwargs['decoder']['l1_linear'] = l1_linear
    net_kwargs['decoder']['add_edge'] = add_edge
    net_kwargs['decoder']['number_of_blocks'] = number_of_blocks
    net_kwargs['decoder']['without_colorattn'] = without_colorattn
    net_kwargs['decoder']['without_colorquery'] = without_colorquery
    net_kwargs['decoder']['without_classification'] = without_classification
    model = create_segmenter(net_kwargs)
    model.to(ptu.device)

    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    if ptu.distributed:
        print('Distributed:', ptu.distributed)
        model = DistributedDataParallel(model, device_ids=[ptu.device],
                                        find_unused_parameters=True)

    # save config
    variant_string = yaml.dump(variant)
    print(f"Configuration:\n{variant_string}")
    variant["net_kwargs"] = net_kwargs
    # variant["dataset_kwargs"] = dataset_kwargs
    log_directory.mkdir(parents=True, exist_ok=True)  # mkdir 库文件名
    with open(log_directory / "variant.yml", "w") as file:
        file.write(variant_string)

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    print(f"Encoder parameters: {number_of_parameters(model_without_ddp.encoder)}")
    print(f"Decoder parameters: {number_of_parameters(model_without_ddp.decoder)}")

    # load imgs.
    image_path = 'example.JPEG'

    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    luminance_resized = rgb_to_lab(image)[:, :, :1]
    ab_resized = rgb_to_lab(image)[:, :, 1:]  # np.float32
    original_lab = luminance_resized[:, :, 0]
    lab = original_lab.reshape((256 * 256))

    mask_luminance = load_mask(mask_luminance_range_number)

    mask_patch_color = np.zeros((256 ** 2, 313),
                                dtype=np.float32)  # [256x256, 313]
    for luminance_range in range(mask_luminance_range_number):
        start_l1, end_l1 = luminance_range * (
                    100 // mask_luminance_range_number), (
                                       luminance_range + 1) * (
                                       100 // mask_luminance_range_number)
        if end_l1 == 100:
            index_l1 = np.where((lab >= start_l1) & (lab <= end_l1))[0]
        else:
            index_l1 = np.where((lab >= start_l1) & (lab < end_l1))[0]
        mask_patch_color[index_l1, :] = mask_luminance[luminance_range, :]

    mask = torch.from_numpy(mask_patch_color)  # [256*256, 313]

    image_luminance = numpy_to_torch(luminance_resized)
    image_ab = numpy_to_torch(ab_resized)
    mask, image_luminance, image_ab = mask.unsqueeze(
        0), image_luminance.unsqueeze(0), image_ab.unsqueeze(0)

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    with torch.no_grad():
        image_luminance = image_luminance.to(ptu.device)
        image_ab = image_ab.to(ptu.device)
        mask = mask.to(ptu.device)
        ab_prediction, query_prediction, query_actual, out_feature = model_without_ddp.inference(
            image_luminance, image_ab, mask, applied=True)
        save_images(image_luminance, image_ab, ab_prediction, "colorized.JPEG",
                    'saved_directory')


def lab_to_rgb(image):
    assert image.dtype == np.float32
    return (255 * np.clip(color.lab2rgb(image), 0, 1)).astype(np.uint8)


def save_images(image_luminance, image_ab, ab_prediction, filenames, directory):
    image_luminance = torch.cat((image_luminance, ab_prediction.detach()),
                                dim=1).cpu()
    batch_size = image_luminance.size(0)
    fake_rgb_list, real_rgb_list, only_rgb_list = [], [], []
    for j in range(batch_size):
        image_lab_numpy = image_luminance[j].numpy().transpose(1, 2, 0)  # np.float32
        image_rgb = lab_to_rgb(image_lab_numpy)  # np.uint8      # [0-255]
        fake_rgb_list.append(image_rgb)

        image_path = os.path.join(directory, filenames)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            skimage.io.imsave(image_path, fake_rgb_list[j].astype(np.uint8))
            print('successful save imgs. ')


if __name__ == '__main__':
    test_func()
