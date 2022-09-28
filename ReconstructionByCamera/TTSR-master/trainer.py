from utils import calculate_peak_signal_to_noise_ratio_and_structural_similarity
from model import Vgg19

import os
import numpy as np
from imageio import imread, imsave
from PIL import Image

import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional
import torch.optim as optim
import torchvision.utils as utils

from option import args
import cv2

import random


class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = torch_neural_network.DataParallel(self.vgg19, list(
                range(self.args.num_gpu)))

        self.params = [
            {
                "params": filter(lambda p: p.requires_grad,
                                 self.model.MainNet.parameters() if
                                 args.num_gpu == 1 else self.model.module.MainNet.parameters()),
                "lr": args.lr_rate
            },
            {
                "params": filter(lambda p: p.requires_grad,
                                 self.model.LearnableTextureExtractor.parameters() if
                                 args.num_gpu == 1 else self.model.module.LearnableTextureExtractor.parameters()),
                "lr": args.lr_rate_lte
            }
        ]
        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2),
                                    eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_peak_signal_to_noise_ratio = 0.
        self.max_peak_signal_to_noise_ratio_epoch = 0
        self.max_structural_similarity = 0.
        self.max_structural_similarity_epoch = 0

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            # model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {keys_: values_ for keys_, values_ in
                                     torch.load(model_path,
                                                map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' % (
            self.optimizer.param_groups[0]['lr']))

        super_resolution = None
        low_resolution_image = None
        high_resolution = None
        high_resolution_images_as_references = None
        down_and_upsampled_Ref_image = None

        random_super_resolution = None
        random_low_resolution_image = None
        random_high_resolution = None
        random_high_resolution_images_as_references = None
        random_down_and_upsampled_Ref_image = None

        get_random = False

        for image_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            low_resolution_image = sample_batched['low_resolution_image']
            upsampled_low_resolution_image = sample_batched[
                'upsampled_low_resolution_image']
            high_resolution = sample_batched['high_resolution']
            high_resolution_images_as_references = sample_batched[
                'high_resolution_images_as_references']
            down_and_upsampled_Ref_image = sample_batched[
                'down_and_upsampled_Ref_image']
            super_resolution, Soft_attention_map, Hard_attention_output_lv3, Hard_attention_output_lv2, Hard_attention_output_lv1 = self.model(
                low_resolution_image=low_resolution_image,
                upsampled_low_resolution_image=upsampled_low_resolution_image,
                high_resolution_images_as_references=high_resolution_images_as_references,
                down_and_upsampled_Ref_image=down_and_upsampled_Ref_image)

            ### calc loss
            is_print = ((
                                    image_batch + 1) % self.args.print_every == 0)  ### flag of print

            reconstruction_loss = self.args.rec_w * self.loss_all[
                'reconstruction_loss'](super_resolution, high_resolution)
            loss = reconstruction_loss
            if (is_print):
                self.logger.info(('init ' if is_init else '') + 'epoch: ' + str(
                    current_epoch) +
                                 '\t batch: ' + str(image_batch + 1))
                self.logger.info(
                    'reconstruction_loss: %.10f' % (reconstruction_loss.item()))

            if (not is_init):
                if ('perceptual_loss_SR_HR' in self.loss_all):
                    super_resolution_relu5_1 = self.vgg19(
                        (super_resolution + 1.) / 2.)
                    with torch.no_grad():
                        high_resolution_relu5_1 = self.vgg19(
                            (high_resolution.detach() + 1.) / 2.)
                    perceptual_loss_SR_HR = self.args.per_w * self.loss_all[
                        'perceptual_loss_SR_HR'](super_resolution_relu5_1,
                                                 high_resolution_relu5_1)
                    loss += perceptual_loss_SR_HR
                    if (is_print):
                        self.logger.info('perceptual_loss_SR_HR: %.10f' % (
                            perceptual_loss_SR_HR.item()))
                if ('perceptual_loss_SR_T' in self.loss_all):
                    super_resolution_lv1, super_resolution_lv2, super_resolution_lv3 = self.model(
                        super_resolution=super_resolution)
                    perceptual_loss_SR_T = self.args.tpl_w * self.loss_all[
                        'perceptual_loss_SR_T'](super_resolution_lv3,
                                                super_resolution_lv2,
                                                super_resolution_lv1,
                                                Soft_attention_map,
                                                Hard_attention_output_lv3,
                                                Hard_attention_output_lv2,
                                                Hard_attention_output_lv1)
                    loss += perceptual_loss_SR_T
                    if (is_print):
                        self.logger.info('perceptual_loss_SR_T: %.10f' % (
                            perceptual_loss_SR_T.item()))
                if ('adversarial_loss' in self.loss_all):
                    adversarial_loss = self.args.adv_w * self.loss_all[
                        'adversarial_loss'](super_resolution, high_resolution)
                    loss += adversarial_loss
                    if (is_print):
                        self.logger.info('adversarial_loss: %.10f' % (
                            adversarial_loss.item()))

            # ramdom get an image during training
            if random.randrange(0, 100, 1) < 50:
                random_super_resolution = super_resolution
                random_low_resolution_image = low_resolution_image
                random_high_resolution = high_resolution
                random_high_resolution_images_as_references = high_resolution_images_as_references
                random_down_and_upsampled_Ref_image = down_and_upsampled_Ref_image
                get_random = True

            loss.backward()
            self.optimizer.step()

        '''
        output preview images
        '''
        if not get_random:
            random_super_resolution = super_resolution
            random_low_resolution_image = low_resolution_image
            random_high_resolution = high_resolution
            random_high_resolution_images_as_references = high_resolution_images_as_references
            random_down_and_upsampled_Ref_image = down_and_upsampled_Ref_image

        with torch.no_grad():
            super_resolution_save = (random_super_resolution + 1.) * 127.5
            super_resolution_save = np.transpose(
                super_resolution_save.squeeze().round().cpu().numpy(),
                (1, 2, 0)).astype(np.uint8)
            super_resolution_save = cv2.resize(super_resolution_save,
                                               (160,160))

            low_resolution_image_save = (
                                                    random_low_resolution_image + 1.) * 127.5
            low_resolution_image_save = np.transpose(
                low_resolution_image_save.squeeze().round().cpu().numpy(),
                (1, 2, 0)).astype(np.uint8)
            low_resolution_image_save = cv2.resize(low_resolution_image_save,
                                                   (160,160))

            high_resolution_save = (random_high_resolution + 1.) * 127.5
            high_resolution_save = np.transpose(
                high_resolution_save.squeeze().round().cpu().numpy(),
                (1, 2, 0)).astype(np.uint8)
            high_resolution_save = cv2.resize(high_resolution_save,
                                              (160,160))

            high_resolution_images_as_references_save = (
                                                                    random_high_resolution_images_as_references + 1.) * 127.5
            high_resolution_images_as_references_save = np.transpose(
                high_resolution_images_as_references_save.squeeze().round().cpu().numpy(),
                (1, 2, 0)).astype(np.uint8)
            high_resolution_images_as_references_save = cv2.resize(
                high_resolution_images_as_references_save,
                (160,160))

            down_and_upsampled_Ref_image_save = (
                                                            random_down_and_upsampled_Ref_image + 1.) * 127.5
            down_and_upsampled_Ref_image_save = np.transpose(
                down_and_upsampled_Ref_image_save.squeeze().round().cpu().numpy(),
                (1, 2, 0)).astype(np.uint8)
            down_and_upsampled_Ref_image_save = cv2.resize(
                down_and_upsampled_Ref_image_save,
                (160,160))

            preview_image = np.hstack((high_resolution_images_as_references_save, high_resolution_save, super_resolution_save))

            preview_path_file = os.path.join(self.args.preview_directory,
                                             str(current_epoch) + '.jpg')

            imsave(preview_path_file, preview_image)

        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.', ''): tmp[key] for key in
                                tmp if
                                (('SearchNet' not in key) and (
                                        '_copy' not in key))}
            model_name = self.args.save_dir.strip('/') + '/model/model_' + str(
                current_epoch).zfill(5) + '.pt'
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        self.logger.info(
            'Epoch ' + str(current_epoch) + ' evaluation process...')

        if (self.args.dataset == 'CUFED'):
            self.model.eval()
            with torch.no_grad():
                peak_signal_to_noise_ratio, structural_similarity, count = 0., 0., 0.
                for image_batch, sample_batched in enumerate(
                        self.dataloader['test']['1']):
                    count += 1
                    sample_batched = self.prepare(sample_batched)
                    low_resolution_image = sample_batched[
                        'low_resolution_image']
                    upsampled_low_resolution_image = sample_batched[
                        'upsampled_low_resolution_image']
                    high_resolution = sample_batched['high_resolution']
                    high_resolution_images_as_references = sample_batched[
                        'high_resolution_images_as_references']
                    down_and_upsampled_Ref_image = sample_batched[
                        'down_and_upsampled_Ref_image']

                    super_resolution, _, _, _, _ = self.model(
                        low_resolution_image=low_resolution_image,
                        upsampled_low_resolution_image=upsampled_low_resolution_image,
                        high_resolution_images_as_references=high_resolution_images_as_references,
                        down_and_upsampled_Ref_image=down_and_upsampled_Ref_image)
                    if (self.args.eval_save_results):
                        super_resolution_save = (super_resolution + 1.) * 127.5
                        super_resolution_save = np.transpose(
                            super_resolution_save.squeeze().round().cpu().numpy(),
                            (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results',
                                            str(image_batch).zfill(5) + '.png'),
                               super_resolution_save)

                    ### calculate Peak_Signal_to_Noise_Ratio and Structural_SIMilarity
                    _peak_signal_to_noise_ratio, _structural_similarity = calculate_peak_signal_to_noise_ratio_and_structural_similarity(
                        super_resolution.detach(),
                        high_resolution.detach())

                    peak_signal_to_noise_ratio += _peak_signal_to_noise_ratio
                    structural_similarity += _structural_similarity

                peak_signal_to_noise_ratio_ave = peak_signal_to_noise_ratio / count
                structural_similarity_ave = structural_similarity / count
                self.logger.info(
                    'high_resolution_images_as_references  peak_signal_to_noise_ratio (now): %.3f \t structural_similarity (now): %.4f' % (
                        peak_signal_to_noise_ratio_ave,
                        structural_similarity_ave))
                if (
                        peak_signal_to_noise_ratio_ave > self.max_peak_signal_to_noise_ratio):
                    self.max_peak_signal_to_noise_ratio = peak_signal_to_noise_ratio_ave
                    self.max_peak_signal_to_noise_ratio_epoch = current_epoch
                if (structural_similarity_ave > self.max_structural_similarity):
                    self.max_structural_similarity = structural_similarity_ave
                    self.max_structural_similarity_epoch = current_epoch
                self.logger.info(
                    'high_resolution_images_as_references  peak_signal_to_noise_ratio (max): %.3f (%d) \t structural_similarity (max): %.4f (%d)'
                    % (self.max_peak_signal_to_noise_ratio,
                       self.max_peak_signal_to_noise_ratio_epoch,
                       self.max_structural_similarity,
                       self.max_structural_similarity_epoch))

        self.logger.info('Evaluation over.')

    def test(self):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' % (self.args.lr_path))
        self.logger.info('ref path:    %s' % (self.args.ref_path))

        ### LR and LR_sr
        low_resolution_image = imread(self.args.lr_path, as_gray=False,
                                      pilmode='RGB')
        height_1, width_1 = low_resolution_image.shape[:2]
        upsampled_low_resolution_image = np.array(
            Image.fromarray(low_resolution_image).resize(
                (width_1 * 4, height_1 * 4), Image.BICUBIC))

        ### Ref and Ref_sr
        high_resolution_images_as_references = imread(self.args.ref_path,
                                                      as_gray=False,
                                                      pilmode='RGB')
        height_2, width_2 = high_resolution_images_as_references.shape[:2]
        height_2, width_2 = height_2 // 4 * 4, width_2 // 4 * 4
        high_resolution_images_as_references = high_resolution_images_as_references[
                                               :height_2, :width_2, :]
        down_and_upsampled_Ref_image = np.array(
            Image.fromarray(high_resolution_images_as_references).resize(
                (width_2 // 4, height_2 // 4), Image.BICUBIC))
        down_and_upsampled_Ref_image = np.array(
            Image.fromarray(down_and_upsampled_Ref_image).resize(
                (width_2, height_2), Image.BICUBIC))

        ### change type
        low_resolution_image = low_resolution_image.astype(np.float32)
        upsampled_low_resolution_image = upsampled_low_resolution_image.astype(
            np.float32)
        high_resolution_images_as_references = high_resolution_images_as_references.astype(
            np.float32)
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image.astype(
            np.float32)

        ### rgb range to [-1, 1]
        low_resolution_image = low_resolution_image / 127.5 - 1.
        upsampled_low_resolution_image = upsampled_low_resolution_image / 127.5 - 1.
        high_resolution_images_as_references = high_resolution_images_as_references / 127.5 - 1.
        down_and_upsampled_Ref_image = down_and_upsampled_Ref_image / 127.5 - 1.

        ### to tensor
        low_resolution_image_to_tensor = torch.from_numpy(
            low_resolution_image.transpose((2, 0, 1))).unsqueeze(0).float().to(
            self.device)
        upsampled_low_resolution_image_to_tensor = torch.from_numpy(
            upsampled_low_resolution_image.transpose((2, 0, 1))).unsqueeze(
            0).float().to(self.device)
        high_resolution_images_as_references_to_tensor = torch.from_numpy(
            high_resolution_images_as_references.transpose(
                (2, 0, 1))).unsqueeze(0).float().to(self.device)
        down_and_upsampled_Ref_image_to_tensor = torch.from_numpy(
            down_and_upsampled_Ref_image.transpose((2, 0, 1))).unsqueeze(
            0).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            super_resolution, _, _, _, _ = self.model(
                low_resolution_image=low_resolution_image_to_tensor,
                upsampled_low_resolution_image=upsampled_low_resolution_image_to_tensor,
                high_resolution_images_as_references=high_resolution_images_as_references_to_tensor,
                down_and_upsampled_Ref_image=down_and_upsampled_Ref_image_to_tensor)
            super_resolution_save = (super_resolution + 1.) * 127.5
            super_resolution_save = np.transpose(
                super_resolution_save.squeeze().round().cpu().numpy(),
                (1, 2, 0)).astype(np.uint8)
            save_path = os.path.join(self.args.save_dir, 'save_results',
                                     os.path.basename(self.args.lr_path))
            '''
            custom output size
            '''
            if args.custom_size:
                super_resolution_save = cv2.resize(super_resolution_save,
                                                   (args.image_width,
                                                    args.image_height))

            imsave(save_path, super_resolution_save)
            self.logger.info('output path: %s' % (save_path))

        self.logger.info('Test over.')
