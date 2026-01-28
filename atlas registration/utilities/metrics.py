#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method collection to evaluate the method performance.

"""
import torch
from monai.metrics import SSIMMetric, compute_average_surface_distance
from torch.nn.functional import one_hot

def dice_similarity_coefficient(prediction, target, num_classes = 10, epsilon=1e-8):
    """
    Computes the Dice coefficient between prediction and target.

    Parameters
    :param prediction: tensor of shape (B,H, W)
    :param target: tensor of shape (B,H, W)
    :param num_classes: number of classes
    :param epsilon: smoothing factor to prevent 0/0
    :return: dice coefficient between prediction and target
    """
    assert prediction.shape == target.shape
    assert prediction.dim() == 3

    if num_classes > 1:
        target = one_hot(target, num_classes).permute(0, 3, 1, 2)
        prediction = one_hot(prediction, num_classes).permute(0, 3, 1, 2)
    else:
        target = target.squeeze()
        prediction = prediction.squeeze()

    axes = (-1, -2)
    intersection = (prediction * target).sum(axis=axes)
    area_sum = prediction.sum(axis=axes) + target.sum(axis=axes)

    return (2 * intersection + epsilon) / (area_sum + epsilon)

def sim_metrics(warped_img, warped_label, target_img, target_label):
    """
    Method to compute similarity metrics between prediction and target.
    :param warped_img: tensor of shape (b,h,w)
    :param warped_label: tensor of shape (b,h,w)
    :param target_img: tensor of shape (b,h,w)
    :param target_label: tensor of shape (b,h,w)
    :return: dictionary of metrics
    """
    ssim_metric = SSIMMetric(spatial_dims=2)

    one_hot_mov = one_hot(warped_label.type(torch.LongTensor), 10).permute(0, 3, 1, 2)
    one_hot_fix = one_hot(target_label.type(torch.LongTensor), 10).permute(0, 3, 1, 2)

    ssim = ssim_metric(warped_img.unsqueeze(1), target_img.unsqueeze(1)).mean()
    assd = compute_average_surface_distance(one_hot_mov.squeeze(), one_hot_fix.squeeze(), include_background=False, spacing=[6/496,6/512], symmetric=True).mean(dim=0) * 1000
    dice = dice_similarity_coefficient(warped_label.type(torch.LongTensor), target_label.type(torch.LongTensor), num_classes=10).mean(dim=0)

    return {'ssim' : ssim, 'assd': assd, 'dice': dice}


def deformation_field_metrics(deformation, gird_coors):
    jaco = compute_jacobian_matrix(gird_coors, deformation)
    det = torch.det(jaco)
    neg_jac_det_perc = (det < 0).sum() / det.shape[0] *100

    return {'neg_jac_det_perc' : neg_jac_det_perc, 'jac_det': det}


def compute_jacobian_matrix(input_coords, output, add_identity=True):
    """Compute the Jacobian matrix."""

    jacobian_matrix = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        jacobian_matrix[:, i, :] = gradient(input_coords, output[:, i])
        if add_identity:
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])
    return jacobian_matrix


def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True,allow_unused=True
    )[0]
    return grad
