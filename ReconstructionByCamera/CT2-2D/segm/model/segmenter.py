import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.autograd import Function
from segm.model.utils import padding, unpadding, SoftEncodeAB, CIELAB, AnnealedMeanDecodeQuery
from timm.models.layers import trunc_normal_
import time


class GetClassWeights:
    def __init__(self, cielab, lambda_=0.5, device='cuda'):
        prior = torch.from_numpy(cielab.gamut.prior)

        uniform = torch.zeros_like(prior)
        uniform[prior > 0] = 1 / (prior > 0).sum().type_as(uniform)

        self.weights = 1 / ((1 - lambda_) * prior + lambda_ * uniform)
        self.weights /= torch.sum(prior * self.weights)

    def __call__(self, ab_actual):
        return self.weights[ab_actual.argmax(dim=1, keepdim=True)].to(ab_actual.device)


class RebalanceLoss(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(context, data_input, weights):
        context.save_for_backward(weights)

        return data_input.clone()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(context, gradient_output):
        weights, = context.saved_tensors

        # reweigh gradient pixelwise so that rare colors get a chance to
        # contribute
        gradient_input = gradient_output * weights

        # second return value is None since we are not interested in the
        # gradient with respect to the weights
        return gradient_input, None


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        number_of_colors,
        before_classify,
        backbone,
        without_classification,
    ):
        super().__init__()
        self.number_of_colors = number_of_colors
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.default_cielab = CIELAB()
        self.encode_ab = SoftEncodeAB(self.default_cielab)
        self.decode_query = AnnealedMeanDecodeQuery(self.default_cielab, special_T=0.38)
        self.class_rebal_lambda = 0.5
        self.get_class_weights = GetClassWeights(self.default_cielab,
                                                 lambda_=self.class_rebal_lambda)
        self.before_classify = before_classify
        self.backbone = backbone
        self.without_classification = without_classification
        self.rebalance_loss = RebalanceLoss.apply

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        no_weight_decay_parameters = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return no_weight_decay_parameters


    def normalize_luminance(self, luminance, to):
        # follow Real-Time/ CIC
        normalized = (luminance - 50) / 100.
        return normalized

    @torch.cuda.amp.autocast()
    def forward(self, luminance, ground_truth_ab, input_mask=None):
        image = self.normalize_luminance(luminance, (-1, 1))  # [-1, 1]

        image = image.repeat(1, 3, 1, 1)
        height, width = image.size(2), image.size(3)
        image_information, image_information_position = self.encoder(image, return_features=True)      #x: [BS, N, D]
        if 'vit' in self.backbone:
            # remove CLS/DIST tokens for decoding
            number_of_extra_tokens = 1 + self.encoder.distilled
            image_information = image_information[:, number_of_extra_tokens:]
        if image_information_position is not None:
            image_information_position = image_information_position[:, number_of_extra_tokens:]
        masks, out_feat = self.decoder(image_information, (height, width), input_mask, image_information_position, image)     # [b, 313, H/P, W/P], out_feat: [B, 2, h, w]

        if not self.without_classification:
            query_prediction = masks      # multi-scaled, [B, 313, 256, 256]
            query_actual = self.encode_ab(ground_truth_ab)
            # rebalancing
            color_weights = self.get_class_weights(query_actual)
            query_prediction = self.rebalance_loss(query_prediction, color_weights)
            ab_prediction = self.decode_query(query_prediction)  # softmax to [0, 1]
        else:
            ab_prediction, query_prediction, query_actual = None, None, None
        return ab_prediction, query_prediction, query_actual, out_feat

    def inference(self, luminance, image_ab, input_mask=None, applied=False):
        image = self.normalize_luminance(luminance, (-1, 1))       # [-1, 1]
        image = image.repeat(1, 3, 1, 1)
        height, width = image.size(2), image.size(3)

        image_information, image_information_position = self.encoder(image, return_features=True)  # x: [BS, N, D]
        if 'vit' in self.backbone:
            # remove CLS/DIST tokens for decoding
            number_of_extra_tokens = 1 + self.encoder.distilled
            image_information = image_information[:, number_of_extra_tokens:]
        if image_information_position is not None:
            image_information_position = image_information_position[:, number_of_extra_tokens:]
        masks, out_feat = self.decoder(image_information, (height, width), input_mask, image_information_position, image)  # [b, K, H/P, W/P]
        if not self.without_classification:
            query_prediction = masks      # [B, 313, 256, 256]
            ab_prediction = self.decode_query(query_prediction, applied=applied)     # softmax to [0, 1]
            query_actual = self.encode_ab(image_ab)
        else:
            ab_prediction, query_prediction, query_actual = None, None, None

        return ab_prediction, query_prediction, query_actual, out_feat

    def convert_ab(self, image_ab):
        query_actual = self.encode_ab(image_ab)
        ab_actual = self.decode_query(query_actual, is_actual=True)
        return ab_actual

    def get_attention_map_enc(self, image, layer_id):
        return self.encoder.get_attention_map(image, layer_id)

    def get_attention_map_dec(self, image, layer_id):
        image_information = self.encoder(image, return_features=True)

        # remove CLS/DIST tokens for decoding
        number_of_extra_tokens = 1 + self.encoder.distilled
        image_information = image_information[:, number_of_extra_tokens:]

        return self.decoder.get_attention_map(image_information, layer_id)
