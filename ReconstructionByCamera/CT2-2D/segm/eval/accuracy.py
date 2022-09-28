import click
import torch

import segm.utils.torch as ptu

from segm.utils.logger import MetricLogger

from segm.model.factory import create_vit
from segm.data.factory import create_dataset
from segm.data.utils import STATS
from segm.metrics import accuracy
from segm import config


def compute_labels(model, batch):
    image = batch["im"]
    target = batch["target"]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model.forward(image)
    accuracy1, accuracy5 = accuracy(output, target, topk=(1, 5))

    return accuracy1.item(), accuracy5.item()


def evaluation_dataset(model, dataset_kwargs):
    created_dataset = create_dataset(dataset_kwargs)
    print_frequency = 20
    header = ""
    logger = MetricLogger(delimiter="  ")

    for batch in logger.log_every(created_dataset, print_frequency, header):
        for key_, value_ in batch.items():
            batch[key_] = value_.to(ptu.device)
        accuracy1, accuracy5 = compute_labels(model, batch)
        batch_size = batch["im"].size(0)
        logger.update(acc1=accuracy1, n=batch_size)
        logger.update(acc5=accuracy5, n=batch_size)
    print(f"Imagenet accuracy: {logger}")


@click.command()
@click.argument("backbone", type=str)
@click.option("--imagenet-dir", type=str)
@click.option("-bs", "--batch-size", default=32, type=int)
@click.option("-nw", "--num-workers", default=10, type=int)
@click.option("-gpu", "--gpu/--no-gpu", default=True, is_flag=True)
def main(backbone, imagenet_directory, batch_size, number_of_workers, gpu):
    ptu.set_gpu_mode(gpu)
    loaded_config = config.load_config()
    loaded_config = loaded_config["model"][backbone]
    loaded_config["backbone"] = backbone
    loaded_config["image_size"] = (loaded_config["image_size"], loaded_config["image_size"])

    dataset_kwargs = dict(
        dataset="imagenet",
        root_dir=imagenet_directory,
        image_size=loaded_config["image_size"],
        crop_size=loaded_config["image_size"],
        patch_size=loaded_config["patch_size"],
        batch_size=batch_size,
        number_of_workers=number_of_workers,
        split="val",
        normalization=STATS[loaded_config["normalization"]],
    )

    model = create_vit(loaded_config)
    model.to(ptu.device)
    model.eval()
    evaluation_dataset(model, dataset_kwargs)


if __name__ == "__main__":
    main()
