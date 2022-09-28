import sys
import click
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import shutil

import torch
import torch.nn.functional as torch_functional
from torch.nn.parallel import DistributedDataParallel as DDP

from segm.utils import distributed
from segm.utils.logger import MetricLogger
import segm.utils.torch as ptu

from segm.model.factory import load_model
from segm.data.factory import create_dataset
from segm.metrics import gather_data, compute_metrics

from segm.model.utils import inference
from segm.data.utils import segmentation_to_rgb, rgb_denormalize, IGNORE_LABEL
from segm import config


def blend_image(image, segmentation, alpha=0.5):
    pillow_image = Image.fromarray(image)
    pillow_segmentation = Image.fromarray(segmentation)
    image_blend = Image.blend(pillow_image, pillow_segmentation, alpha).convert("RGB")
    return np.asarray(image_blend)


def save_image(save_directory, save_name, image, segmentation_prediction, segmentation_ground_truth, colors, blend, normalization):
    segmentation_rgb = segmentation_to_rgb(segmentation_ground_truth[None], colors)
    prediction_rgb = segmentation_to_rgb(segmentation_prediction[None], colors)
    image_unnormalization = rgb_denormalize(image, normalization)
    save_directory = Path(save_directory)

    # save images
    image_uint = (image_unnormalization.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
    segmentation_rgb_uint = (255 * segmentation_rgb.cpu().numpy()).astype(np.uint8)
    segmentation_prediction_uint = (255 * prediction_rgb.cpu().numpy()).astype(np.uint8)
    for i in range(prediction_rgb.shape[0]):
        if blend:
            blend_prediction = blend_image(image_uint[i], segmentation_prediction_uint[i])
            blend_ground_truth = blend_image(image_uint[i], segmentation_rgb_uint[i])
            images = (image_uint[i], blend_prediction, blend_ground_truth)
        else:
            images = (image_uint[i], segmentation_prediction_uint[i], segmentation_rgb_uint[i])
        for image, image_directory in zip(
            images,
            (save_directory / "input", save_directory / "prediction", save_directory / "gt"),
        ):
            pillow_out = Image.fromarray(image)
            image_directory.mkdir(exist_ok=True)
            pillow_out.save(image_directory / save_name)


def process_batch(
    model,
    batch,
    window_size,
    window_stride,
    window_batch_size,
):
    images = batch["im"]
    images_metas = batch["im_metas"]
    original_shape = images_metas[0]["ori_shape"]
    original_shape = (original_shape[0].item(), original_shape[1].item())
    filename = batch["im_metas"][0]["ori_filename"][0]

    model_without_ddp = model
    if ptu.distributed:
        model_without_ddp = model.module
    segmentation_prediction = inference(
        model_without_ddp,
        images,
        images_metas,
        original_shape,
        window_size,
        window_stride,
        window_batch_size,
    )
    segmentation_prediction = segmentation_prediction.argmax(0)
    image = torch_functional.interpolate(images[-1], original_shape, mode="bilinear")

    return filename, image.cpu(), segmentation_prediction.cpu()


def evaluation_dataset(
    model,
    multiscale,
    model_directory,
    blend,
    window_size,
    window_stride,
    window_batch_size,
    save_images,
    frac_dataset,
    dataset_kwargs,
):
    created_dataset = create_dataset(dataset_kwargs)
    normalization = created_dataset.dataset.norm
    dataset_name = dataset_kwargs["dataset"]
    image_size = dataset_kwargs["image_size"]
    cat_names = created_dataset.base_dataset.names
    number_of_colors = created_dataset.unwrapped.number_of_colors
    if multiscale:
        created_dataset.dataset.set_multiscale_mode()

    logger = MetricLogger(delimiter="  ")
    header = ""
    print_frequency = 50

    images = {}
    segmentation_prediction_maps = {}
    index = 0
    for batch in logger.log_every(created_dataset, print_frequency, header):
        colors = batch["colors"]
        filename, image, segmentation_prediction = process_batch(
            model,
            batch,
            window_size,
            window_stride,
            window_batch_size,
        )
        images[filename] = image
        segmentation_prediction_maps[filename] = segmentation_prediction
        index += 1
        if index > len(created_dataset) * frac_dataset:
            break

    segmentation_ground_truth_maps = created_dataset.dataset.get_ground_truth_segmentation_maps()
    if save_images:
        save_directory = model_directory / "images"
        if ptu.dist_rank == 0:
            if save_directory.exists():
                shutil.rmtree(save_directory)
            save_directory.mkdir()
        if ptu.distributed:
            torch.distributed.barrier()

        for name in sorted(images):
            instance_directory = save_directory
            filename = name

            if dataset_name == "cityscapes":
                filename_list = name.split("/")
                instance_directory = instance_directory / filename_list[0]
                filename = filename_list[-1]
                if not instance_directory.exists():
                    instance_directory.mkdir()

            save_image(
                instance_directory,
                filename,
                images[name],
                segmentation_prediction_maps[name],
                torch.tensor(segmentation_ground_truth_maps[name]),
                colors,
                blend,
                normalization,
            )
        if ptu.dist_rank == 0:
            shutil.make_archive(save_directory, "zip", save_directory)
            # shutil.rmtree(save_directory)
            print(f"Saved eval images in {save_directory}.zip")

    if ptu.distributed:
        torch.distributed.barrier()
        segmentation_prediction_maps = gather_data(segmentation_prediction_maps)

    scores = compute_metrics(
        segmentation_prediction_maps,
        segmentation_ground_truth_maps,
        number_of_colors,
        ignore_index=IGNORE_LABEL,
        resolution_enhancement_technology_concat_iou=True,
        distributed=ptu.distributed,
    )

    if ptu.dist_rank == 0:
        scores["inference"] = "single_scale" if not multiscale else "multi_scale"
        suffix = "ss" if not multiscale else "ms"
        scores["concat_iou"] = np.round(100 * scores["concat_iou"], 2).tolist()
        for key_, value_ in scores.items():
            if key_ != "concat_iou" and key_ != "inference":
                scores[key_] = value_.item()
            if key_ != "concat_iou":
                print(f"{key_}: {scores[key_]}")
        scores_str = yaml.dump(scores)
        with open(model_directory / f"scores_{suffix}.yml", "w") as file:
            file.write(scores_str)


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--image-size", default=None, type=int)
@click.option("--multiscale/--singlescale", default=False, is_flag=True)
@click.option("--blend/--no-blend", default=True, is_flag=True)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--window-batch-size", default=4, type=int)
@click.option("--save-images/--no-save-images", default=False, is_flag=True)
@click.option("-frac-dataset", "--frac-dataset", default=1.0, type=float)
def main(
    model_path,
    dataset_name,
    image_size,
    multiscale,
    blend,
    window_size,
    window_stride,
    window_batch_size,
    save_images,
    frac_dataset,
):

    model_directory = Path(model_path).parent

    # start distributed mode
    ptu.set_gpu_mode(True)
    distributed.initial_process()

    model, variant = load_model(model_path)
    patch_size = model.patch_size
    model.eval()
    model.to(ptu.device)
    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    loaded_config = config.load_config()
    dataset_config = loaded_config["dataset"][dataset_name]
    normalization = variant["dataset_kwargs"]["normalization"]
    if image_size is None:
        image_size = dataset_config.get("image_size", variant["dataset_kwargs"]["image_size"])
    if window_size is None:
        window_size = dataset_config.get(
            "window_size", variant["dataset_kwargs"]["crop_size"]
        )
    if window_stride is None:
        window_stride = dataset_config.get(
            "window_stride", variant["dataset_kwargs"]["crop_size"]
        )

    dataset_kwargs = dict(
        dataset=dataset_name,
        image_size=image_size,
        crop_size=image_size,
        patch_size=patch_size,
        batch_size=1,
        number_of_workers=10,
        split="val",
        normalization=normalization,
        crop=False,
        rep_aug=False,
    )

    evaluation_dataset(
        model,
        multiscale,
        model_directory,
        blend,
        window_size,
        window_stride,
        window_batch_size,
        save_images,
        frac_dataset,
        dataset_kwargs,
    )

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    main()
