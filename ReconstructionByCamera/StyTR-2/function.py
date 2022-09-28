import torch


def calculate_mean_standard(feature, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feature.size()
    assert (len(size) == 4)
    number, channal = size[:2]
    feature_variable = feature.view(number, channal, -1).var(dim=2) + eps
    feature_standard = feature_variable.sqrt().view(number, channal, 1, 1)
    feature_mean = feature.view(number, channal, -1).mean(dim=2).view(number, channal,
                                                                   1, 1)
    return feature_mean, feature_standard


def calculate_mean_standard1(feature, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feature.size()
    # assert (len(size) == 4)
    # WH,N, C = size
    feature_variable = feature.var(dim=0) + eps
    feature_standard = feature_variable.sqrt()
    feature_mean = feature.mean(dim=0)
    return feature_mean, feature_standard


def normal(feature, eps=1e-5):
    feature_mean, feature_standard = calculate_mean_standard(feature, eps)
    normalized = (feature - feature_mean) / feature_standard
    return normalized


def normal_style(feature, eps=1e-5):
    feature_mean, feature_standard = calculate_mean_standard1(feature, eps)
    normalized = (feature - feature_mean) / feature_standard
    return normalized


def _calculate_feature_flatten_mean_standard(feature):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feature.size()[0] == 3)
    assert (isinstance(feature, torch.FloatTensor))
    feature_flatten = feature.view(3, -1)
    mean = feature_flatten.mean(dim=-1, keepdim=True)
    standard = feature_flatten.std(dim=-1, keepdim=True)
    return feature_flatten, mean, standard


def _mat_sqrt(need_svd):
    U, S, V = torch.svd(need_svd)
    # SVD把一个矩阵分解成U、S、V三个矩阵。其中U、V是两个列向量彼此正交的矩阵。S是除了对角线元素其他都为0的对角矩阵。
    return torch.mm(torch.mm(U, S.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_flatten, source_flatten_mean, source_flattten_standard = _calculate_feature_flatten_mean_standard(
        source)
    source_flatten_normalization = (
                                           source_flatten - source_flatten_mean.expand_as(
                                       source_flatten)) / source_flattten_standard.expand_as(
        source_flatten)
    source_flatten_covariance_eye = \
        torch.mm(source_flatten_normalization,
                 source_flatten_normalization.t()) + torch.eye(3)

    target_flatten, target_flatten_mean, target_flatten_standard = _calculate_feature_flatten_mean_standard(
        target)
    target_flatten_normalization = (
                                           target_flatten - target_flatten_mean.expand_as(
                                       target_flatten)) / target_flatten_standard.expand_as(
        target_flatten)
    target_flatten_covariance_eye = \
        torch.mm(target_flatten_normalization,
                 target_flatten_normalization.t()) + torch.eye(3)

    source_flatten_normalization_transfer = torch.mm(
        _mat_sqrt(target_flatten_covariance_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_flatten_covariance_eye)),
                 source_flatten_normalization)
    )

    source_flatten_transfer = (source_flatten_normalization_transfer *
                               target_flatten_standard.expand_as(
                                   source_flatten_normalization) +
                               target_flatten_mean.expand_as(
                                   source_flatten_normalization))

    return source_flatten_transfer.view(source.size())
