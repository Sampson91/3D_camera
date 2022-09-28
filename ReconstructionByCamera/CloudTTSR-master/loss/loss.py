from loss import discriminator

import torch
import torch.nn as torch_neural_network
import torch.nn.functional as torch_neural_network_functional
import torch.optim as optim


class ReconstructionLoss(torch_neural_network.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = torch_neural_network.L1Loss()
        elif (type == 'l2'):
            self.loss = torch_neural_network.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, super_resolution, high_resolution):
        return self.loss(super_resolution, high_resolution)


class PerceptualLoss(torch_neural_network.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, super_resolution_relu5_1, high_resolution_relu5_1):
        loss = torch_neural_network_functional.mse_loss(
            super_resolution_relu5_1, high_resolution_relu5_1)
        return loss


class TPerceptualLoss(torch_neural_network.Module):
    def __init__(self, use_S=True, type='l2'):
        super(TPerceptualLoss, self).__init__()
        self.use_Soft_attention_map = use_S
        self.type = type

    def gram_matrix(self, map):
        batch, channel, height, width = map.size()
        map_size = map.view(batch, channel, height * width)
        map_size_Transpose = map_size.transpose(1, 2)
        Gram = map_size.bmm(map_size_Transpose) / (height * width * channel)
        return Gram

    def forward(self, map_lv3, map_lv2, map_lv1, Soft_attention_map,
                Hard_attention_output_lv3, Hard_attention_output_lv2,
                Hard_attention_output_lv1):
        # S.size(): [N, 1, h, w]
        if (self.use_Soft_attention_map):
            Soft_attention_map_lv3 = torch.sigmoid(Soft_attention_map)
            Soft_attention_map_lv2 = torch.sigmoid(
                torch_neural_network_functional.interpolate(Soft_attention_map,
                                                            size=(
                                                                Soft_attention_map.size(
                                                                    -2) * 2,
                                                                Soft_attention_map.size(
                                                                    -1) * 2),
                                                            mode='bicubic'))
            Soft_attention_map_lv1 = torch.sigmoid(
                torch_neural_network_functional.interpolate(Soft_attention_map,
                                                            size=(
                                                                Soft_attention_map.size(
                                                                    -2) * 4,
                                                                Soft_attention_map.size(
                                                                    -1) * 4),
                                                            mode='bicubic'))
        else:
            Soft_attention_map_lv3, Soft_attention_map_lv2, Soft_attention_map_lv1 = 1., 1., 1.

        if (self.type == 'l1'):
            loss_texture = torch_neural_network_functional.l1_loss(
                map_lv3 * Soft_attention_map_lv3,
                Hard_attention_output_lv3 * Soft_attention_map_lv3)
            loss_texture += torch_neural_network_functional.l1_loss(
                map_lv2 * Soft_attention_map_lv2,
                Hard_attention_output_lv2 * Soft_attention_map_lv2)
            loss_texture += torch_neural_network_functional.l1_loss(
                map_lv1 * Soft_attention_map_lv1,
                Hard_attention_output_lv1 * Soft_attention_map_lv1)
            loss_texture /= 3.
        elif (self.type == 'l2'):
            loss_texture = torch_neural_network_functional.mse_loss(
                map_lv3 * Soft_attention_map_lv3,
                Hard_attention_output_lv3 * Soft_attention_map_lv3)
            loss_texture += torch_neural_network_functional.mse_loss(
                map_lv2 * Soft_attention_map_lv2,
                Hard_attention_output_lv2 * Soft_attention_map_lv2)
            loss_texture += torch_neural_network_functional.mse_loss(
                map_lv1 * Soft_attention_map_lv1,
                Hard_attention_output_lv1 * Soft_attention_map_lv1)
            loss_texture /= 3.

        return loss_texture


class AdversarialLoss(torch_neural_network.Module):
    def __init__(self, logger, use_cpu=False, num_gpu=1, gan_type='WGAN_GP',
                 gan_k=1,
                 lr_dis=1e-4, train_crop_size=200064):

        super(AdversarialLoss, self).__init__()
        self.logger = logger
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = discriminator.Discriminator(
            train_crop_size * 4).to(self.device)
        if (num_gpu > 1):
            self.discriminator = torch_neural_network.DataParallel(
                self.discriminator, list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)

        # if (D_path):
        #     self.logger.info('load_D_path: ' + D_path)
        #     D_state_dict = torch.load(D_path)
        #     self.discriminator.load_state_dict(D_state_dict['D'])
        #     self.optimizer.load_state_dict(D_state_dict['D_optim'])

    def forward(self, fake, real):
        fake_detach = fake.detach()
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            discriminator_fake = self.discriminator(fake_detach)
            discriminator_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_discriminator = (
                        discriminator_fake - discriminator_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    # WGAN_GP_hat = fack.mul(1 - epsilon) + real.mul(epsilon)
                    WGAN_GP_hat = fake_detach.mul(1 - epsilon) + real.mul(
                        epsilon)
                    WGAN_GP_hat.requires_grad = True
                    discriminator_hat = self.discriminator(WGAN_GP_hat)
                    gradients = torch.autograd.grad(
                        outputs=discriminator_hat.sum(), inputs=WGAN_GP_hat,
                        retain_graph=True, create_graph=True, only_inputs=True)[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_discriminator += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(discriminator_real),
                                          valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(discriminator_fake),
                                          fake_score)
                loss_discriminator = (real_loss + fake_loss) / 2.

            # Discriminator update
            loss_discriminator.backward()
            self.optimizer.step()

        discriminator_fake_for_generator = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_generator = -discriminator_fake_for_generator.mean()
        elif (self.gan_type == 'GAN'):
            loss_generator = self.bce_loss(
                torch.sigmoid(discriminator_fake_for_generator), valid_score)

        # Generator loss
        return loss_generator

    def state_dict(self):
        Discriminator_state_dict = self.discriminator.state_dict()
        Discriminator_optimizer_state_dict = self.optimizer.state_dict()
        return Discriminator_state_dict, Discriminator_optimizer_state_dict


def get_loss_dict(args, logger):
    loss = {}
    if (abs(args.rec_w - 0) <= 1e-8):
        raise SystemExit('NotImplementError: ReconstructionLoss must exist!')
    else:
        loss['reconstruction_loss'] = ReconstructionLoss(type='l1')
    if (abs(args.per_w - 0) > 1e-8):
        loss['perceptual_loss_SR_HR'] = PerceptualLoss()
    if (abs(args.tpl_w - 0) > 1e-8):
        loss['perceptual_loss_SR_T'] = TPerceptualLoss(use_S=args.tpl_use_S,
                                                       type=args.tpl_type)
    if (abs(args.adv_w - 0) > 1e-8):
        loss['adversarial_loss'] = AdversarialLoss(logger=logger,
                                                   use_cpu=args.cpu,
                                                   num_gpu=args.num_gpu,
                                                   gan_type=args.GAN_type,
                                                   gan_k=args.GAN_k,
                                                   lr_dis=args.lr_rate_dis,
                                                   train_crop_size=args.train_crop_size)
    return loss