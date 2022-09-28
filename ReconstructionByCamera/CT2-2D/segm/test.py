import sys
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
import re
from tqdm import tqdm


def check_required_folders(save_directory):
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)


@click.command(help="")
@click.option("--log-directory", type=str, help="logging directory")
@click.option("--dataset", default='random', type=str)  # origin: coco
@click.option('--dataset_directory', default='', type=str)
@click.option("--image-size", default=16 * 15, type=int,
              help="dataset resize size")  # 256 patch size==16 have to be n*16
@click.option("--crop-size", default=16 * 15, type=int)  # 256 == 16*16
@click.option("--window-size", default=16 * 15, type=int)  # 256 == 16*16
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="vit_large_patch16_384",
              type=str)  # try this, and freeze first several blocks.  origin: vit_tiny_patch16_384
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
@click.option('--add_mask', type=bool, default=False)  # valid original: True
@click.option('--partial_finetune', type=bool,
              default=False)  # compare validation, last finetune all blocks.
@click.option('--add_l1_loss', type=bool,
              default=True)  # add after classification.
@click.option('--l1_weight', type=float, default=10)
@click.option('--color_position', type=bool,
              default=True)  # add color position in color token.
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
@click.option('--number_of_layers', type=int, default=2)
@click.option('--without_colorattn', type=bool, default=False)
@click.option('--without_colorquery', type=bool, default=False)
@click.option('--without_classification', type=bool, default=False)
@click.option('--mask_random', type=bool, default=False)
@click.option('--color_token_number', type=int, default=313)
@click.option('--sin_color_position', type=bool, default=False)
@click.option('--save_directory', type=str, default='segm/data/CT2_output',
              help="logging directory")
def main(
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
        number_of_layers,
        without_colorattn,
        without_colorquery,
        without_classification,
        mask_random,
        color_token_number,
        sin_color_position,
        save_directory,
):
    check_required_folders(save_directory=save_directory)
    # start distributed mode
    ptu.set_gpu_mode(True, local_rank)
    # distributed.init_process()

    # parapeter only
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    torch.distributed.init_process_group(backend="gloo")

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
            mask_random=mask_random,
            number_of_colors=color_token_number,
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
    # checkpoint_path = log_directory / 'checkpoint.pth'  # base

    dataset_kwargs = variant["dataset_kwargs"]
    validation_kwargs = dataset_kwargs.copy()
    validation_kwargs["split"] = "val"
    validation_kwargs["batch_size"] = 1  # val_batch_size = = 1
    validation_loader = create_dataset(validation_kwargs)

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["number_of_colors"] = color_token_number
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
    net_kwargs['decoder']['number_of_layers'] = number_of_layers
    net_kwargs['decoder']['without_colorattn'] = without_colorattn
    net_kwargs['decoder']['without_colorquery'] = without_colorquery
    net_kwargs['decoder']['without_classification'] = without_classification
    net_kwargs['decoder']['sin_color_position'] = sin_color_position
    model = create_segmenter(net_kwargs)
    model.to(ptu.device)
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume:
        checkpoint_files = os.listdir(log_directory)
        checkpoint_exist = False
        checkpoint_path_file = None
        i = 0
        for checkpoint_file_ in checkpoint_files:
            file_name, ext = os.path.splitext(checkpoint_file_)
            if ext == '.pth':

                checkpoint_file_epoch = re.findall("\d+\.?\d*",
                                                   file_name)  # 提取文件名中的数字
                checkpoint_file_epoch = int(checkpoint_file_epoch[0])

                i += 1
                if i > 1:
                    checkpoint_path_file_name, ext = os.path.splitext(
                        checkpoint_path_file)
                    checkpoint_path_file_epoch = re.findall("\d+\.?\d*",
                                                            checkpoint_path_file_name)  # 提取文件名中的数字
                    checkpoint_path_file_epoch = int(
                        checkpoint_path_file_epoch[0])
                    if checkpoint_path_file_epoch < checkpoint_file_epoch:
                        checkpoint_path_file = checkpoint_file_
                else:
                    checkpoint_path_file = checkpoint_file_
                checkpoint_path = os.path.join(log_directory,
                                               checkpoint_path_file)

            if i:
                checkpoint_exist = True
    if resume and checkpoint_exist:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print('checkpoint:', checkpoint)
        model.load_state_dict(checkpoint["model"])

    if ptu.distributed:
        print('Distributed:', ptu.distributed)
        model = DistributedDataParallel(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_string = yaml.dump(variant)
    print(f"Configuration:\n{variant_string}")
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_directory.mkdir(parents=True, exist_ok=True)
    with open(log_directory / "variant.yml", "w") as file:
        file.write(variant_string)

    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    number_of_epochs = variant["algorithm_kwargs"]["number_of_epochs"]
    evaluation_frequency = variant["algorithm_kwargs"]["evaluation_frequency"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    print(f"Val dataset length: {len(validation_loader.dataset)}")
    print(f"Encoder parameters: {number_of_parameters(model_without_ddp.encoder)}")
    print(f"Decoder parameters: {number_of_parameters(model_without_ddp.decoder)}")

    for epoch in tqdm(range(start_epoch, number_of_epochs)):
        # evaluate
        evaluation_epoch = epoch % evaluation_frequency == 0 or epoch == number_of_epochs - 1
        if evaluation_epoch:
            evaluation_logger = evaluate(
                epoch,
                model,
                validation_loader,
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
                log_directory,
                save_directory=save_directory
            )
            print(f"Stats [{epoch}]:", evaluation_logger, flush=True)

    distributed.barrier()
    distributed.destroy_process()
    # sys.exit(1)


if __name__ == "__main__":
    main()
