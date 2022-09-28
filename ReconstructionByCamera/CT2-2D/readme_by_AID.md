# recommendation: run by command prompt
# pip install requirements
cd to CT2-2D directory  
`pip install -r requirements`
# conda install requirements
`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
# run by Command Prompt
Run the comment lines in segm/train.sh and segm/test.sh or README.md
# run by parameters
Delete 'segm/' in data_dir in config.yml for both random and coco at the bottom or manually create a directory.
# train dataset directory
segm/data/train
# saving checkpoint method
1. default: save_ssd = True: save the first epoch and the last epoch; otherwise, save every 5 epochs.
2. use‘--save_ssd = False’: save each epoch.
# save numbers of checkpoint
Because a checkpoint needs about 3.5GB, too many checkpoints will need a lot of disk space.     
1. default: save_space = True: only keep the recent 5 checkpoints.
2. use '--save_space = False': keep all checkpoints.
# test dataset directory
segm/data/test
# test image output directory
segm/data/CT2_output
# checkpoint directory
1. run by prompt: '.pth' files in segm/vit-large
2. run by parameters: '.pth' files in segm/segm/vit-large
# vit pretrained model
'.npz' files in segm/resources
vit pretrained model need to be downloaded by following the link:   
<https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz>

# preview during training
from left to right      
rgb, gray, train output