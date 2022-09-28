from torch.utils.data import DataLoader
from importlib import import_module


def getdataloader(args):
    ### import module
    import_dataset = import_module('dataset.' + args.dataset.lower())

    if (args.dataset == 'CUFED'):
        data_train = getattr(import_dataset, 'TrainSet')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)
        dataloader_test = {}
        # for i in range(5):
        data_test = getattr(import_dataset, 'TestSet')(args)
        dataloader_test = DataLoader(data_test, batch_size=1, num_workers=args.num_workers)
        # data_test = getattr(import_dataset, 'TestSet')(args=args, high_resolution_images_as_references_level=str(i + 1))
        # dataloader_test[str(i + 1)] = DataLoader(data_test, batch_size=1,
        #                                              shuffle=False,
        #                                              num_workers=args.num_workers)
        data_infer = getattr(import_dataset, "InferenceSet")(args)
        dataloader_infer = DataLoader(data_infer, batch_size=1, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test, 'infer': dataloader_infer}
        # dataloader = {'train': dataloader_train, 'test': dataloader_test}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader
