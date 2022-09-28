from option import args
from utils import mkExpDir
from dataset import dataloader
from model import TextureTransformerSuperResolution
from loss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import torch.nn as torch_neural_network
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # make save_dir
    _logger = mkExpDir(args)

    # dataloader of training set and testing set
    _dataloader = dataloader.getdataloader(args) if (not args.test) else None

    # device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = TextureTransformerSuperResolution.TextureTransformerSuperResolution(
        args).to(device)
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = torch_neural_network.DataParallel(_model,
                                                   list(range(args.num_gpu)))

    # loss
    _loss_all = get_loss_dict(args, _logger)

    # trainer
    trainer = Trainer(args, _logger, _dataloader, _model, _loss_all)

    # test / eval / train
    if (args.test):
        trainer.load(model_path=args.model_path)
        trainer.test()
    elif (args.eval):
        trainer.load(model_path=args.model_path)
        trainer.evaluate()
    elif (args.infer):
        trainer.load(model_path=args.model_path)
        trainer.infer()
    else:
        print("num_init_epochs: ", args.num_init_epochs)
        for epoch in range(1, 2):  # args.num_init_epochs + 1):
            trainer.train(current_epoch=epoch, is_init=True)
        # temp_is_init = False
        print("num_init_epochs: ", args.num_epochs)
        for epoch in range(1, args.num_epochs + 1):
            trainer.train(epoch, False)
            # below evaluate process should be add sometimes
            if (epoch % args.val_every == 0):
                trainer.evaluate(current_epoch=epoch)
