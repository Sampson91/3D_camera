import segm.utils.torch as ptu

from segm.data import COCODataset, RandomDataset
from segm.data import Loader


def create_dataset(dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    number_of_workers = dataset_kwargs.pop("number_of_workers")
    split = dataset_kwargs.pop("split")

    if dataset_name == 'coco':
        dataset = COCODataset(split=split, **dataset_kwargs)
    elif dataset_name == 'random':
        dataset = RandomDataset(**dataset_kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

    dataset = Loader(
        dataset=dataset,
        batch_size=batch_size,
        number_of_workers=number_of_workers,
        distributed=ptu.distributed,
        split=split,
    )
    return dataset
