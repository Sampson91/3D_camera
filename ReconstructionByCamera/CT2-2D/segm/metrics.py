import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as torch_functional
import os
import pickle as pkl
from pathlib import Path
import tempfile
import shutil
from mmseg.core import mean_iou
from PIL import Image
from skimage import measure
# from skimage.measure import compare_ssim, compare_psnr
# from skimage.measure import structural_similarity as compare_ssim
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
from torchvision.transforms import transforms
from scipy import linalg
from torchvision.models.inception import inception_v3
from torch.autograd import Variable
from scipy.stats import entropy
import torch.utils.data
from skimage import color, io

import torch.distributed as distribution

import segm.utils.torch as ptu

"""
ImageNet classifcation accuracy
"""


def accuracy(output, target, topk=(1,)):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)  # 前k个prediction中的最大值
        batch_size = target.size(0)

        _, prediction = output.topk(maxk, 1, True,
                                    True)  # pred: return indices.(the class of the max value.)
        prediction = prediction.t()
        correct = prediction.eq(target.view(1, -1).expand_as(prediction))

        resized_k = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k /= batch_size
            resized_k.append(correct_k)
        return resized_k


def classify_accuracy(qurey_prediction, query_actual, topk=(1,)):
    # print('q_pred.size', q_pred.size(), q_actual.size())    # [b, 313, h, w]
    batch_size, number_of_colors, height, width = qurey_prediction.shape[0], \
                                                  qurey_prediction.shape[1], \
                                                  qurey_prediction.shape[2], \
                                                  qurey_prediction.shape[3]
    with torch.no_grad():
        maxk = max(topk)
        _, prediction = qurey_prediction.detach().topk(maxk, 1, True,
                                                       True)  # [b, 5, h, w]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(
            batch_size * height * width, maxk)  # [b * h *w, 5]
        # print(pred.device, q_pred.device)
        prediction_label = torch.zeros(batch_size * height * width,
                                       number_of_colors).to(
            prediction.device).scatter(1, prediction, 5)

        _, actual = query_actual.detach().topk(maxk, 1, True,
                                               True)  # [b, 5, h, w]
        actual = actual.permute(0, 2, 3, 1).contiguous().view(
            batch_size * height * width, maxk)
        actual_label = torch.ones(batch_size * height * width,
                                  number_of_colors).to(actual.device).scatter(1,
                                                                              actual,
                                                                              5)

        correct = prediction_label.eq(actual_label)
        correct_numbers = torch.sum(correct)
        accuracy = correct_numbers / (batch_size * maxk * height * width)
        return accuracy.item()


"""
Segmentation mean IoU
based on collect_results_cpu
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/apis/test.py#L160-L200
"""


def gather_data(segmentaion_prediction, temporary_directory=None):
    """
    distributed data gathering
    prediction and ground truth are stored in a common tmp directory
    and loaded on the master node to compute metrics
    """
    if temporary_directory is None:
        tmpprefix = os.path.expandvars("$WORK/temp")
    else:
        tmpprefix = temporary_directory
    MAX_LEN = 512
    # 32 is whitespace
    directory_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8,
                                  device=ptu.device)
    if ptu.dist_rank == 0:
        temporarydirectory = tempfile.mkdtemp(prefix=tmpprefix)
        temporarydirectory = torch.tensor(
            bytearray(temporarydirectory.encode()), dtype=torch.uint8,
            device=ptu.device
        )
        directory_tensor[: len(temporarydirectory)] = temporarydirectory
    # broadcast tmpdir from 0 to to the other nodes
    distribution.broadcast(directory_tensor, 0)
    temporarydirectory = directory_tensor.cpu().numpy().tobytes().decode().rstrip()
    temporarydirectory = Path(temporarydirectory)
    """
    Save results in temp file and load them on main process
    """
    temporary_file = temporarydirectory / f"part_{ptu.dist_rank}.pkl"
    pkl.dump(segmentaion_prediction, open(temporary_file, "wb"))
    distribution.barrier()
    segmentaion_prediction = {}
    if ptu.dist_rank == 0:
        for i in range(ptu.world_size):
            part_segmentaion_prediction = pkl.load(
                open(temporarydirectory / f"part_{i}.pkl", "rb"))
            segmentaion_prediction.update(part_segmentaion_prediction)
        shutil.rmtree(temporarydirectory)
    return segmentaion_prediction


def compute_metrics(
        segmentation_prediction,
        segmentation_ground_truth,
        number_of_colors,
        ignore_index=None,
        resolution_enhancement_technology_concat_iou=False,
        temporary_directory=None,
        distributed=False,
):
    resolution_enhancement_technology_metrics_mean = torch.zeros(3, dtype=float,
                                                                 device=ptu.device)
    if ptu.dist_rank == 0:
        list_segmentation_prediction = []
        list_segmentation_ground_truth = []
        keys = sorted(segmentation_prediction.keys())
        for k in keys:
            list_segmentation_prediction.append(
                np.asarray(segmentation_prediction[k]))
            list_segmentation_ground_truth.append(
                np.asarray(segmentation_ground_truth[k]))
        resolution_enhancement_technology_metrics = mean_iou(
            results=list_segmentation_prediction,
            gt_seg_maps=list_segmentation_ground_truth,
            num_classes=number_of_colors,
            ignore_index=ignore_index,
        )
        resolution_enhancement_technology_metrics = [
            resolution_enhancement_technology_metrics["aAcc"],
            resolution_enhancement_technology_metrics["Acc"],
            resolution_enhancement_technology_metrics["IoU"]]
        resolution_enhancement_technology_metrics_mean = torch.tensor(
            [
                np.round(np.nanmean(
                    resolution_enhancement_technology_metric.astype(
                        np.float)) * 100, 2)
                for resolution_enhancement_technology_metric in
                resolution_enhancement_technology_metrics
            ],
            dtype=float,
            device=ptu.device,
        )
        concat_iou = resolution_enhancement_technology_metrics[2]
    # broadcast metrics from 0 to all nodes
    if distributed:
        distribution.broadcast(resolution_enhancement_technology_metrics_mean,
                               0)
    pix_acc, mean_acc, miou = resolution_enhancement_technology_metrics_mean
    resolution_enhancement_technology = dict(pixel_accuracy=pix_acc,
                                             mean_accuracy=mean_acc,
                                             mean_iou=miou)
    if resolution_enhancement_technology_concat_iou and ptu.dist_rank == 0:
        resolution_enhancement_technology["concat_iou"] = concat_iou
    return resolution_enhancement_technology


####################################################################################
###################### Evalutaion Model #############################
def lab_to_rgb(image):
    assert image.dtype == np.float32
    return (255 * np.clip(color.lab2rgb(image), 0, 1)).astype(np.uint8)


def rgb_to_lab(image):
    assert image.dtype == np.uint8
    return color.rgb2lab(image).astype(np.float32)


class INCEPTION_V3(nn.Module):
    def __init__(self, inception_state_dictionary):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        self.model.load_state_dict(inception_state_dictionary)
        for parameter_ in self.model.parameters():
            parameter_.requires_grad = False
            # requires_grad torch tensor 的库

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        inception_v3_neural_network = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        inception_v3_neural_network[:, 0] = (inception_v3_neural_network[:,
                                             0] - 0.485) / 0.229
        inception_v3_neural_network[:, 1] = (inception_v3_neural_network[:,
                                             1] - 0.456) / 0.224
        inception_v3_neural_network[:, 2] = (inception_v3_neural_network[:,
                                             2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        inception_v3_neural_network = torch_functional.interpolate(
            inception_v3_neural_network, size=(299, 299), mode='bilinear',
            align_corners=True)
        # 299 x 299 x 3
        inception_v3_neural_network = self.model(inception_v3_neural_network)
        inception_v3_neural_network = nn.Softmax(dim=-1)(
            inception_v3_neural_network)
        return inception_v3_neural_network


class InceptionV3FrechetInceptionDistance(nn.Module):
    # Frechet Inception Distance == FID
    """pretrained InceptionV3 network returning feature maps"""
    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIMENSION = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 inception_state_dictionary,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        """
        super(InceptionV3FrechetInceptionDistance, self).__init__()

        self.resize_input = resize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3()
        inception.load_state_dict(inception_state_dictionary)
        # load_state_dict torch module 的库
        for parameter_ in inception.parameters():
            parameter_.requires_grad = False

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

    def forward(self, input_parameters):
        """Get Inception feature maps
        Parameters
        ----------
        input_parameters : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        out_parameters = []
        inception_v3_feature_map = input_parameters

        if self.resize_input:
            inception_v3_feature_map = torch_functional.interpolate(
                inception_v3_feature_map, size=(299, 299),
                mode='bilinear')

        inception_v3_feature_map = inception_v3_feature_map.clone()
        # [-1.0, 1.0] --> [0, 1.0]
        inception_v3_feature_map = inception_v3_feature_map * 0.5 + 0.5
        inception_v3_feature_map[:, 0] = inception_v3_feature_map[:, 0] * (
                0.229 / 0.5) + (0.485 - 0.5) / 0.5
        inception_v3_feature_map[:, 1] = inception_v3_feature_map[:, 1] * (
                0.224 / 0.5) + (0.456 - 0.5) / 0.5
        inception_v3_feature_map[:, 2] = inception_v3_feature_map[:, 2] * (
                0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for index, block in enumerate(self.blocks):
            inception_v3_feature_map = block(inception_v3_feature_map)
            if index in self.output_blocks:
                out_parameters.append(inception_v3_feature_map)

            if index == self.last_needed_block:
                break

        return out_parameters


def get_activations(images, model, batch_size, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # d0 = images.shape[0]
    image_size_0 = int(images.size(0))
    if batch_size > image_size_0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = image_size_0

    number_of_batches = image_size_0 // batch_size
    number_of_used_images = number_of_batches * batch_size

    prediction_array = np.empty((number_of_used_images, 2048))
    for i in range(number_of_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, number_of_batches),
                  end='',
                  flush=True)
        start = i * batch_size
        end = start + batch_size

        '''batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if loaded_config.CUDA:
            batch = batch.cuda()'''
        batch = images[start:end]

        prediction = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if prediction.shape[2] != 1 or prediction.shape[3] != 1:
            prediction = torch_functional.adaptive_avg_pool2d(prediction,
                                                              output_size=(
                                                                  1, 1))

        prediction_array[start:end] = prediction.cpu().data.numpy().reshape(
            batch_size,
            -1)

    if verbose:
        print(' done')

    return prediction_array


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    difference = mu1 - mu2

    # Product might be almost singular
    covariance_mean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covariance_mean).all():
        message = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
        print(message)
        offset = np.eye(sigma1.shape[0]) * eps
        covariance_mean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3):
            maximum = np.max(np.abs(covariance_mean.imag))
            raise ValueError('Imaginary component {}'.format(maximum))
        covariance_mean = covariance_mean.real

    trace_covariance_mean = np.trace(covariance_mean)

    return (difference.dot(difference) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * trace_covariance_mean)


def calculate_peak_signal_to_noise_ratio_numpy(image1, image2):
    # Peak signal-to-noise ratio == psnr
    import numpy as np
    squared_error_map = (1. * image1 - image2) ** 2
    current_mean_squared_error = np.mean(squared_error_map)
    return 20 * np.log10(255. / np.sqrt(current_mean_squared_error))


def average_structure_similarity_index_measure_peak_signal_to_noise_ratio(
        predicted_directory, ground_truth_directory):
    image_names = os.listdir(ground_truth_directory)
    image_counts = 0
    total_structure_similarity_index_measure, total_peak_signal_to_noise_ratio = 0, 0
    # Structure Similarity Index Measure = ssim
    total_structure_similarity_index_measure_convert, total_peak_signal_to_noise_ratio_convert = 0, 0
    for image in image_names:
        print('calculate ssim/psnr', image_counts)
        ground_truth_path = os.path.join(ground_truth_directory, image)
        ground_truth = np.array(
            Image.open(ground_truth_path).convert('RGB')).astype(np.uint8)
        # need gt: rgb-> lab-> rgb
        ground_truth_lab = rgb_to_lab(ground_truth)
        ground_truth_rgb = lab_to_rgb(ground_truth_lab)

        prediction_path = os.path.join(predicted_directory, 'fake_' + image)
        predicted = np.array(Image.open(prediction_path).convert("RGB"))
        structure_similarity_index_measure = measure.compare_ssim(ground_truth,
                                                                  predicted,
                                                                  data_range=225,
                                                                  multichannel=True)
        structure_similarity_index_measure_convert = measure.compare_ssim(
            ground_truth_rgb, predicted, data_range=225,
            multichannel=True)
        peak_signal_to_noise_ratio = measure.compare_psnr(ground_truth,
                                                          predicted, 255)
        peak_signal_to_noise_ratio_convert = measure.compare_psnr(
            ground_truth_rgb, predicted, 255)
        total_structure_similarity_index_measure += structure_similarity_index_measure
        total_peak_signal_to_noise_ratio += peak_signal_to_noise_ratio
        total_structure_similarity_index_measure_convert += structure_similarity_index_measure_convert
        total_peak_signal_to_noise_ratio_convert += peak_signal_to_noise_ratio_convert
        image_counts += 1
    assert image_counts == 5000
    structure_similarity_index_measure_average = total_structure_similarity_index_measure / image_counts
    peak_signal_to_noise_ratio_avg = total_peak_signal_to_noise_ratio / image_counts
    structure_similarity_index_measure_average_convert = total_structure_similarity_index_measure_convert / image_counts
    peak_signal_to_noise_ratio_average_convert = total_peak_signal_to_noise_ratio_convert / image_counts
    return (
        structure_similarity_index_measure_average,
        peak_signal_to_noise_ratio_avg,
        structure_similarity_index_measure_average_convert,
        peak_signal_to_noise_ratio_average_convert)


def average_learned_perceptual_image_patch_similarity(predicted_directory,
                                                      ground_truth_directory):
    # Learned Perceptual Image Patch Similarity == lpips
    image_names = os.listdir(ground_truth_directory)
    image_counts = 0
    total_learned_perceptual_image_patch_similarity, total_learned_perceptual_image_patch_similarity_convert = 0, 0
    loss_function_vgg = lpips.LPIPS(net='vgg').cuda()
    # lipips 是库名
    with torch.no_grad():
        for image in image_names:
            print('calculate lpips', image_counts)
            ground_truth_path = os.path.join(ground_truth_directory, image)
            ground_truth = np.array(
                Image.open(ground_truth_path).convert("RGB")).astype(np.uint8)
            # gt: convert rgb->lab->rgb
            ground_truth_lab = rgb_to_lab(ground_truth)
            ground_truth_rgb = lab_to_rgb(ground_truth_lab)

            ground_truth = transforms.ToTensor()(ground_truth)
            ground_truth = transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))(
                ground_truth).cuda()
            ground_truth_rgb = transforms.ToTensor()(ground_truth_rgb)
            ground_truth_rgb = transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))(
                ground_truth_rgb).cuda()

            prediction_path = os.path.join(predicted_directory, "fake_" + image)

            prediction = np.array(Image.open(prediction_path).convert("RGB"))
            prediction = transforms.ToTensor()(prediction)
            prediction = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                prediction).cuda()

            learned_perceptual_image_patch_similarity_vgg = loss_function_vgg(
                ground_truth, prediction).cpu()
            learned_perceptual_image_patch_similarity_vgg_convert = loss_function_vgg(
                ground_truth_rgb, prediction).cpu()
            total_learned_perceptual_image_patch_similarity += learned_perceptual_image_patch_similarity_vgg
            total_learned_perceptual_image_patch_similarity_convert += learned_perceptual_image_patch_similarity_vgg_convert
            image_counts += 1
    assert image_counts == 5000
    learned_perceptual_image_patch_similarity_average = total_learned_perceptual_image_patch_similarity / image_counts
    learned_perceptual_image_patch_similarity_average_convert = total_learned_perceptual_image_patch_similarity_convert / image_counts
    return learned_perceptual_image_patch_similarity_average, learned_perceptual_image_patch_similarity_average_convert


def inception_score(images, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    numbers = len(images)

    assert batch_size > 0
    assert numbers > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print(
                "WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(
        dtype)
    inception_model.eval()
    upsampling = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_prediction(predict_score):
        if resize:
            predict_score = upsampling(predict_score)
        predict_score = inception_model(predict_score)
        return torch_functional.softmax(predict_score).data.cpu().numpy()

    # Get predictions
    predictions = np.zeros((numbers, 1000))
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            # print('i', i)
            batch = batch[0]  # batch[0]=pred_img, batch[1]=gt_img.
            batch = batch.type(dtype)
            batch_variable = Variable(batch)
            batch_size_i = batch.size()[0]

            predictions[
            i * batch_size:i * batch_size + batch_size_i] = get_prediction(
                batch_variable)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = predictions[
               k * (numbers // splits): (k + 1) * (numbers // splits), :]
        part_mean = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            part_mean_i = part[i, :]
            scores.append(entropy(part_mean_i, part_mean))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class visual_stimulus_intent_person_dataset(torch.utils.data.Dataset):
    # Visual stimulus, Intent, Person == vip
    def __init__(self, data_directory, ground_truth_directory):
        # self.gt_dir = '/userhome/SUN_text2img/ImageNet/val'
        self.ground_truth_directory = ground_truth_directory
        self.data_directory = data_directory
        ground_truth_names = os.listdir(self.ground_truth_directory)
        self.filenames = ground_truth_names
        self.transform_list = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      (0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image_directory = os.path.join(self.ground_truth_directory, image_path)
        image = Image.open(image_directory).convert('RGB')

        # convert gt.
        image_numpy = np.array(image).astype(np.uint8)
        image_lab = rgb_to_lab(image_numpy)
        image_rgb = lab_to_rgb(image_lab)

        ground_truth_image = self.transform_list(image).float()
        ground_truth_image_convert = self.transform_list(image_rgb).float()

        fake_directory = os.path.join(self.data_directory, 'fake_' + image_path)

        fake_image = Image.open(fake_directory).convert("RGB")
        fake_image = self.transform_list(fake_image).float()

        return fake_image, ground_truth_image, ground_truth_image_convert

    def __len__(self):
        return len(self.filenames)


def calculate_is(prediction_directory):
    is_mean, is_std = inception_score(
        visual_stimulus_intent_person_dataset(prediction_directory), cuda=True,
        batch_size=32, resize=True, splits=10)
    return is_mean, is_std


def calculate_frechet_inception_distance(prediction_directory,
                                         ground_truth_directory):
    batch_size = 1
    new_batch_size = 1
    inception_path = os.path.join('resources',
                                  "inception_v3_google-1a9a5a14.pth")

    inception_state_dictionary = torch.load(inception_path,
                                            map_location=lambda storage,
                                                                loc: storage)
    block_index = InceptionV3FrechetInceptionDistance.BLOCK_INDEX_BY_DIMENSION[
        2048]
    inception_model_frechet_inception_distance = InceptionV3FrechetInceptionDistance(
        inception_state_dictionary, [block_index])
    inception_model_frechet_inception_distance.cuda()
    inception_model_frechet_inception_distance.eval()
    dataset = visual_stimulus_intent_person_dataset(prediction_directory,
                                                    ground_truth_directory)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False)
    fake_acts_set, acts_set, acts_set_convert = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            prediction, ground_truth, ground_truth_convert = batch[0].cuda(), \
                                                             batch[1].cuda(), \
                                                             batch[2].cuda()
            fake_act = get_activations(prediction,
                                       inception_model_frechet_inception_distance,
                                       new_batch_size)
            real_act = get_activations(ground_truth,
                                       inception_model_frechet_inception_distance,
                                       new_batch_size)
            real_act_convert = get_activations(ground_truth_convert,
                                               inception_model_frechet_inception_distance,
                                               new_batch_size)
            fake_acts_set.append(fake_act)
            acts_set.append(real_act)
            acts_set_convert.append(real_act_convert)
            # break
        acts_set = np.concatenate(acts_set, 0)
        fake_acts_set = np.concatenate(fake_acts_set, 0)
        acts_set_convert = np.concatenate(acts_set_convert, 0)

        real_mu, real_sigma = calculate_activation_statistics(acts_set)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
        real_mu_convert, real_sigma_convert = calculate_activation_statistics(
            acts_set_convert)
        frechet_inception_distance_score = calculate_frechet_distance(real_mu,
                                                                      real_sigma,
                                                                      fake_mu,
                                                                      fake_sigma)
        frechet_inception_distance_score_convert = calculate_frechet_distance(
            real_mu_convert,
            real_sigma_convert,
            fake_mu, fake_sigma)
    return frechet_inception_distance_score, frechet_inception_distance_score_convert
