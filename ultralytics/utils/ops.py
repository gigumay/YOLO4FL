# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import math
import re
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
import random
import torch.nn.functional as F
from types import SimpleNamespace
from typing import OrderedDict
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, box_iou


class Profile(contextlib.ContextDecorator):
    """
    Ultralytics Profile class for timing code execution.

    Use as a decorator with @Profile() or as a context manager with 'with Profile():'. Provides accurate timing
    measurements with CUDA synchronization support for GPU operations.

    Attributes:
        t (float): Accumulated time in seconds.
        device (torch.device): Device used for model inference.
        cuda (bool): Whether CUDA is being used for timing synchronization.

    Examples:
        Use as a context manager to time code execution
        >>> with Profile(device=device) as dt:
        ...     pass  # slow operation here
        >>> print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"

        Use as a decorator to time function execution
        >>> @Profile()
        ... def slow_function():
        ...     time.sleep(0.1)
    """

    def __init__(self, t: float = 0.0, device: Optional[torch.device] = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial accumulated time in seconds.
            device (torch.device, optional): Device used for model inference to enable CUDA synchronization.
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Return a human-readable string representing the accumulated elapsed time."""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """Get current time with CUDA synchronization if applicable."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


def get_obj_boxes(gt_bboxes: torch.Tensor, img_size: tuple = (640, 640), box_padding: float = 0.0):
    """
    Get list of object boxes from gt label tensor. Add padding if specified.
    Args:
        gt_bboxes (torch.Tensor):                   Ground truth bounding boxes with shape (N, num_boxes, 4) in xyxy format.
        img_size (tuple):                           Size of the input image as (height, width).
        box_padding (float):                        Padding factor to apply to bounding boxes (e.g., 0.1 for 10% padding).
    Returns:
        Extracted object features with shape (total_boxes, C, pooled_height, pooled_width). 
    """
    bs, _, _ = gt_bboxes.shape

    if box_padding > 0:
        widths  = gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0]
        heights = gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1]
        dw = widths * box_padding / 2
        dh = heights * box_padding / 2

        gt_bboxes = gt_bboxes.clone()
        gt_bboxes[:, :, 0] -= dw  # x1
        gt_bboxes[:, :, 2] += dw  # x2
        gt_bboxes[:, :, 1] -= dh  # y1
        gt_bboxes[:, :, 3] += dh  # y2
        gt_bboxes = torch.clamp(gt_bboxes, min=0, max=max(img_size))

    box_list = [gt_bboxes[i][(gt_bboxes[i] != 0).any(dim=1)] for i in range(bs)]
    return box_list


def flatten_features(features: torch.Tensor):
    """
    Remove spatial dimensions from features. features must be of shape (N,C,W,H)
    Args:
        features (torch.Tensor): Input features with shape (N, C, W, H).
    Returns:
        Flattened features with shape (N, C).
    """
    features_flattened =  torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
    return  features_flattened


def get_features(embds: list, hyp: SimpleNamespace, gt_bboxes: torch.Tensor=None, msa: torchvision.ops.MultiScaleRoIAlign=None, 
                 use_background: bool=False, all_preds: torch.Tensor=None, all_scores: torch.Tensor=None):
    """
    Extract and flatten features from the neck output feature maps (P3-P5).
    Args:
        embds (list):                                       List of feature maps from the neck with shapes [(N,C1,W1,H1), (N,C2,W2,H2), (N,C3,W3,H3)].
        hyp (SimpleNamespace):                              Hyperparameters including 'isolate_objects' (bool) and 'msa_box_padding' (float).
        gt_bboxes (torch.Tensor, optional):                 Ground truth bounding boxes with shape (N, num_boxes, 4) in xyxy format. Required 
                                                            if 'isolate_objects' is True.
        msa (torchvision.ops.MultiScaleRoIAlign, optional): MultiScaleRoIAlign module for extracting features. Required if 'isolate_objects' is True.
        use_background (bool):                              Whether to extract background features.
        all_preds (torch.Tensor, optional):                 Predicted bounding boxes with shape (bs, num_boxes, 4) in xyxy format. Required if 'use_background' is True.
        all_scores (torch.Tensor, optional):                Confidence scores with shape (bs, num_boxes, 1). Required if 'use_background' is True.
    Returns:
        Extracted and flattened features with shape (total_boxes, C) if 'isolate_objects' is True, otherwise (N, C). Also returns the deficit in empty boxes
        when use_background is True. 
    """

    background_deficit = None
    if not hyp.isolate_objects:
        assert not use_background, "Background prototype extraction not supported for full feature maps!"
        features_3D = embds[0]
    else:
        maps = OrderedDict({f"P{i+3}": fm for i, fm in enumerate(embds)})
        if not use_background:
            box_list = get_obj_boxes(gt_bboxes=gt_bboxes, box_padding=hyp.msa_box_padding)
        else:
            box_list, deficit = sample_empty_boxes(gt_bboxes=gt_bboxes, pred_bboxes=all_preds, pred_scores=all_scores, hn_ratio=hyp.hn_ratio_bckgrnd, imgsz=hyp.imgsz)
            background_deficit = deficit
        
        features_3D = msa.forward(x=maps,  boxes=box_list, image_shapes=[(hyp.imgsz, hyp.imgsz)]*hyp.batch)

    return flatten_features(features_3D), background_deficit


def agg_features(features: torch.Tensor, n_protos: int, is_training: bool =True, clustering_algrthm: KMeans=None):
    """
    Aggregate features into prototypes via mean or clustering.
    Args:
        features (torch.Tensor):                Input features with shape (N, C).
        n_protos (int):                         Number of prototypes (clusters) to extract. 
        is_training (bool):                     Whether the model is in training mode. Clustering is not supported during training.
        clustering_algrthm (KMeans, optional):  Clustering algorithm instance from sklearn. Required if 'n_protos' > 1.
    Returns:
        Aggregated prototypes with shape (n_protos, C).
    """
    if n_protos == 1:
        proto = features.mean(dim=0, keepdim=True)
    else:
        assert not is_training, "Clustering during training currently not supported"
        features_np = features.detach().cpu().numpy()
        clusters = clustering_algrthm.fit(features_np)
        proto = torch.from_numpy(clusters.cluster_centers_).to(device=features.device, dtype=features.dtype)

    assert len(proto.shape) == 2, f"Unexpected output shape: {proto.shape}"    
    return proto


def generate_proto(embds: list, hyp: SimpleNamespace, aggregate: bool, is_training: bool, gt_bboxes: torch.Tensor=None, 
                   msa: torchvision.ops.MultiScaleRoIAlign=None, obj_clustering_algrthm: KMeans=None, use_background: bool=False, 
                   bg_clustering_algrthm: KMeans=None, all_preds: torch.Tensor=None, all_scores: torch.Tensor=None):
    """
    Generate prototypes from neck output feature maps (P3-P5).
    Args:
        embds (list):                                       List of feature maps from the neck with shapes [(N,C1,W1,H1), (N,C2,W2,H2), (N,C3,W3,H3)].
        hyp (SimpleNamespace):                              Hyperparameters including 'isolate_objects' (bool) and 'n_protos' (int).
        aggregate (bool):                                   Whether to aggregate features into prototypes.
        is_training (bool):                                 Whether the model is in training mode. Clustering is not supported during training.
        gt_bboxes (torch.Tensor, optional):                 Ground truth bounding boxes with shape (N, num_boxes, 4) in xyxy format. Required
                                                            if 'isolate_objects' is True.
        msa (torchvision.ops.MultiScaleRoIAlign, optional): MultiScaleRoIAlign module for extracting features. Required if 'isolate_objects' is True.
        obj_clustering_algrthm (KMeans, optional):          Clustering algorithm instance from sklearn for clustering object features. Required if 'n_protos' > 1. 
        use_background (bool):                              Whether to extract background features.
        bg_clustering_algrthm (KMeans, optional):           Clustering algorithm instance from sklearn for clustering background features. Required if 'n_protos' > 1.
        all_preds (torch.Tensor, optional):                 Predicted bounding boxes with shape (bs, num_boxes, 4) in xyxy format. Required if 'use_background' is True.
        all_scores (torch.Tensor, optional):                Confidence scores with shape (bs, num_boxes, 1). Required if 'use_background' is True.
        imgsz (int):                                        Image size.
    Returns:
        Generated prototypes with shape (n_protos, C) if 'aggregate' is True, otherwise (total_boxes, C).
    """
    assert not (use_background and (all_preds is None or all_scores is None)), "'all_preds' and 'all_scores' must be provided when 'use_background' is True"

    features, _ = get_features(embds=embds, hyp=hyp, gt_bboxes=gt_bboxes, msa=msa, use_background=use_background, all_preds=all_preds, all_scores=all_scores)  # (total_boxes, C) or (N, C)

    if not aggregate:
        return features
    else:
        if not use_background:
            return agg_features(features=features, hyp=hyp, is_training=is_training, clustering_algrthm=obj_clustering_algrthm)
        else: 
            return agg_features(features=features, hyp=hyp, is_training=is_training, clustering_algrthm=bg_clustering_algrthm)
    

def assign_local2global_proto(local_proto: torch.Tensor, global_proto: torch.Tensor, return_distances: bool):
    """
    Assign local prototypes to nearest global prototype and return either the corresponding distances or 
    the assigned local prototypes  for each global prototype.
    Args:
        local_proto (torch.Tensor):   Local prototypes with shape (n_local, C).
        global_proto (torch.Tensor):  Global prototypes with shape (n_global, C).
        return_distances (bool):      Whether to return distances to assigned global prototypes or group local prototypes.
    Returns:
        If 'return_distances' is True, returns:
        - assignments (torch.Tensor): Indices of assigned global prototypes for each local prototype with shape (n_local,). 
        - distances (torch.Tensor):   Distances to assigned global prototypes with shape (n_local,).
        If 'return_distances' is False, returns:
        - assignments (torch.Tensor): Indices of assigned global prototypes for each local prototype with shape (n_local,).
        - grouped (dict):             Dictionary mapping global prototype indices to lists of local prototypes assigned to them.
    """

    # Pairwise distances: [n_local, n_global]
    dist_matrix = torch.cdist(local_proto, global_proto, p=2)

    # Nearest global centroid for each local
    assignments = dist_matrix.argmin(dim=1)

    if return_distances:
        # Just the distances to assigned global prototype
        distances = dist_matrix[torch.arange(local_proto.shape[0]), assignments]
        return assignments, distances
    else:
        # Group locals into a list of tensors
        grouped = {idx: [] for idx in range(global_proto.shape[0])}
        for g in range(global_proto.size(0)):
            mask = (assignments == g)
            if mask.any():
                grouped[g].append(local_proto[mask])
    
        return assignments, grouped
    

def compute_cost_matrix(clusters, candidates):
    """
    Compute cost matrix for grouping of prototypes.
    Args:
        clusters (torch.Tensor):   Current clusters with shape (n_clusters, n_protos, C).
        candidates (torch.Tensor):  Candidate prototypes to assign with shape (n_candidates, C).
    Returns:
        Cost matrix with shape (n_candidates, n_clusters).
    """
    cm = torch.zeros((candidates.shape[0], clusters.shape[0]), dtype=torch.float32, device=candidates.device)

    for j in range(clusters.shape[0]):                     
        dists = torch.cdist(candidates, clusters[j], p=2)   
        cm[:, j] = dists.sum(dim=1)              
    
    return cm


def prototype_matching(prototypes, n_orders=10):
    """
    Cluster prototypes into groups of size n_protos by iteratively optimizing via Jonker-Volgenant algorithm 
    to approximat the global cost minimum as measured via the L2 distance.
    Args:
        prototypes (torch.Tensor): Prototypes to cluster with shape (n_prototypes, C).
        n_orders (int):            Number of random orders to try for clustering to reduce order bias.
    Returns:
        best_clusters (torch.Tensor):   Clustered prototypes with shape (n_clusters, n_protos, C).
        best_total_cost (float):        Total cost associated with the best clustering.
    """
    best_total_cost = np.inf
    best_clusters = None
    
    for base_idx in range(prototypes.shape[0]):
        base_set = prototypes[base_idx]
        remaining_indices = [i for i in range(prototypes.shape[0]) if i != base_idx]
        
        for _ in range(n_orders):
            # Shuffle the remaining sets to reduce order bias
            random.shuffle(remaining_indices)
            
            clusters = torch.unsqueeze(base_set, dim=1)
            total_cost = 0.0
            
            for idx in remaining_indices:
                candidate_points = prototypes[idx]
                cost_matrix = compute_cost_matrix(clusters=clusters, candidates=candidate_points)
                row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

                # extend clusters along "points per cluster" axis
                new_points = torch.zeros((clusters.shape[0], 1, clusters.shape[2]), dtype=clusters.dtype, device=clusters.device)
                clusters = torch.cat([clusters, new_points], dim=1)
                
                for r_idx, c_idx in zip(row_ind, col_ind):
                    # insert new point into cluster
                    clusters[c_idx, clusters.shape[1] - 1, :] = candidate_points[r_idx] 
                    total_cost += float(cost_matrix[r_idx, c_idx].item())
            
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_clusters = clusters
    
    return best_clusters, best_total_cost


def sort_preds(pred_bboxes, pred_scores):
    """ 
    Sort predicted boxes by their confidence scores in descending order.
    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes of shape (B, N, 4).
        pred_scores (torch.Tensor): Confidence scores of shape (B, N, 1).
    Returns:
        sorted_bboxes (torch.Tensor): Sorted bounding boxes of shape (B, N, 4).
        sorted_scores (torch.Tensor): Sorted confidence scores of shape (B, N, 1).
    """
    sorted_idx = pred_scores.squeeze(-1).argsort(dim=1, descending=True)
    pred_bboxes = torch.gather(pred_bboxes, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 4))
    pred_scores = torch.gather(pred_scores, 1, sorted_idx.unsqueeze(-1))
    return pred_bboxes, pred_scores


def get_empty_boxes(preds, gt, pred_scores, iou_eps, imgsz):
    """
    Identify predicted boxes that do not overlap with any ground truth boxes (IoU < iou_eps).
    Args:
        preds (torch.Tensor):           Predicted bounding boxes of shape (N, 4).
        gt (torch.Tensor):              Ground truth bounding boxes of shape (M, 4).
        pred_scores (torch.Tensor):     Confidence scores of shape (N, 1).
        iou_eps (float):                IoU threshold to consider a box as empty.
        imgsz(int):                     Image size.
    Returns:
        empty_boxes (torch.Tensor):     Empty bounding boxes of shape (K, 4).
        empty_scores (torch.Tensor):    Confidence scores of empty boxes of shape (K,).
        all_empty_idx (torch.Tensor):   Indices of empty boxes in the original preds tensor of shape (K,).
    """
    if gt.numel() > 0:
        ious = box_iou(preds, gt)
        max_iou, _ = ious.max(dim=1)
        empty_mask = max_iou < iou_eps
    else:
        empty_mask = torch.ones(preds.shape[0], dtype=torch.bool, device=preds.device)

    inside_mask = ((preds >= 0) & (preds <= imgsz)).all(dim=1)
    final_mask = empty_mask & inside_mask

    empty_boxes = preds[final_mask]
    empty_scores = pred_scores[final_mask].squeeze(-1)
    all_empty_idx = torch.arange(empty_boxes.shape[0], device=preds.device)
    return empty_boxes, empty_scores, all_empty_idx


def sample_from_empty(empty_boxes, empty_scores, all_empty_idx, num_gt, hn_ratio):
    """
    Sample empty boxes based on the number of ground truth boxes and hard negative ratio.
    Args:
        empty_boxes (torch.Tensor):     Empty bounding boxes of shape (K, 4).
        empty_scores (torch.Tensor):    Confidence scores of empty boxes of shape (K,).
        all_empty_idx (torch.Tensor):   Indices of empty boxes in the original preds tensor of shape (K,).
        num_gt (int):                   Number of ground truth boxes.
        hn_ratio (float):               Ratio of hard negatives to sample.
    Returns:
        sampled_boxes (torch.Tensor):   Sampled empty bounding boxes of shape (T, 4).
        sel_idx (torch.Tensor):         Indices of sampled boxes in the original empty_boxes tensor of shape (T,).
    """
    available = empty_boxes.shape[0]
    take = min(num_gt, available)

    if take == 0:
        return torch.empty((0, 4), device=empty_boxes.device), torch.tensor([], dtype=torch.long, device=empty_boxes.device)

    hn_count = int(take * hn_ratio)
    rn_count = take - hn_count

    # just to be safe 
    assert len(empty_scores.shape) == 1, "Shape Inconsistency!"

    hn_idx = empty_scores.argsort(descending=True)[:hn_count]

    mask = torch.ones(len(empty_boxes), dtype=torch.bool, device=empty_boxes.device)
    mask[hn_idx] = False
    remaining_idx = all_empty_idx[mask]

    if rn_count > 0 and len(remaining_idx) > 0:
        rn_idx = remaining_idx[torch.randperm(len(remaining_idx))[:rn_count]]
    else:
        rn_idx = torch.tensor([], dtype=torch.long, device=empty_boxes.device)

    sel_idx = torch.cat([hn_idx, rn_idx]) if rn_idx.numel() > 0 else hn_idx
    return empty_boxes[sel_idx], sel_idx


def redistribute_deficit(capacities, total_deficit, empty_boxes_all, empty_scores_all, all_empty_idx_all, empty_boxes_per_img,
                         selected_indices_per_img, hn_ratio):
    """
    Redistribute deficit of empty boxes across images with surplus.
    Args:
        capacities (list):                  List of tuples (b, need, avail) for images with surplus
        total_deficit (int):                Total deficit of empty boxes across the batch.
        empty_boxes_all (list):             List of all empty boxes per image.
        empty_scores_all (list):            List of all empty scores per image.
        all_empty_idx_all (list):           List of all empty indices per image.
        empty_boxes_per_img (list):         Current list of sampled empty boxes per image.
        selected_indices_per_img (list):    Current list of selected indices per image.
        hn_ratio (float):                   Ratio of hard negatives to sample.
    Returns:
        empty_boxes_per_img (list):         Updated list of sampled empty boxes per image after redistribution."""
    if total_deficit <= 0:
        return empty_boxes_per_img

    for b, need, avail in capacities:
        if total_deficit <= 0:
            break

        # capacity = how many more we can take from this image
        extra = avail - need if need > 0 else avail
        if extra <= 0:
            continue

        to_take = min(extra, total_deficit)

        empty_boxes = empty_boxes_all[b]
        empty_scores = empty_scores_all[b]
        all_empty_idx = all_empty_idx_all[b]
        taken = selected_indices_per_img[b]

        if taken.numel() > 0:
            mask = torch.ones(len(empty_boxes), dtype=torch.bool, device=empty_boxes.device)
            mask[taken] = False
            remaining_idx = all_empty_idx[mask]
        else:
            remaining_idx = all_empty_idx

        if len(remaining_idx) > 0:
            remaining_scores = empty_scores[remaining_idx]

            # sort by score first
            sorted_idx = remaining_scores.argsort(descending=True)
            sorted_remaining = remaining_idx[sorted_idx]

            # split into hard negatives + random negatives
            hn_count = int(to_take * hn_ratio)
            rn_count = to_take - hn_count

            hn_idx = sorted_remaining[:hn_count]  # top scores

            remaining_after_hn = sorted_remaining[hn_count:]  # the rest

            if rn_count > 0 and len(remaining_after_hn) > 0:
                rand_perm = torch.randperm(len(remaining_after_hn), device=empty_boxes.device)
                rn_idx = remaining_after_hn[rand_perm[:rn_count]]
            else:
                rn_idx = torch.tensor([], dtype=torch.long, device=empty_boxes.device)

            sel_idx = torch.cat([hn_idx, rn_idx]) if rn_idx.numel() > 0 else hn_idx

            # update outputs
            empty_boxes_per_img[b] = torch.cat([empty_boxes_per_img[b], empty_boxes[sel_idx]], dim=0)
            selected_indices_per_img[b] = torch.cat([selected_indices_per_img[b], sel_idx])
            total_deficit -= to_take

    return empty_boxes_per_img


def sample_empty_boxes(gt_bboxes: torch.Tensor, pred_bboxes: torch.Tensor, pred_scores: torch.Tensor, hn_ratio: float,
                       imgsz: int, iou_eps: float = 1e-5):
    """
    Sample empty bounding boxes for each image in the batch based on ground truth boxes and hard negative ratio.
    Args:
        gt_bboxes (torch.Tensor):    Ground truth bounding boxes of shape (B, M, 4).
        pred_bboxes (torch.Tensor):  Predicted bounding boxes of shape (B, N, 4).
        pred_scores (torch.Tensor):  Confidence scores of shape (B, N, 1).
        hn_ratio (float):            Ratio of hard negatives to sample.
        imgsz (int):                 Image size. 
        iou_eps (float):             IoU threshold to consider a box as empty.
    Returns:
        empty_boxes_per_img (list):  List of sampled empty bounding boxes per image, each of shape (T_i, 4).
    """
    batch_size, _, _ = pred_bboxes.shape

    # Step 1: sort predictions by score
    # pred_bboxes, pred_scores = sort_preds(pred_bboxes=pred_bboxes, pred_scores=pred_scores)

    # holders
    empty_boxes_per_img = [[] for _ in range(batch_size)]
    selected_indices_per_img = [[] for _ in range(batch_size)]
    empty_boxes_all, empty_scores_all, all_empty_idx_all = [], [], []
    deficits, capacities = [], []

    # Process each image
    for b in range(batch_size):
        gt = gt_bboxes[b]
        gt = gt[gt.sum(dim=1) > 0]  # remove padded zeros
        num_gt = gt.shape[0]

        # full empties
        empty_boxes, empty_scores, all_empty_idx = get_empty_boxes(preds=pred_bboxes[b], gt=gt, 
                                                                   pred_scores=pred_scores[b], imgsz=imgsz,
                                                                   iou_eps=iou_eps)

        # sampled empties
        sampled_boxes, sel_idx = sample_from_empty(empty_boxes=empty_boxes, empty_scores=empty_scores, 
                                                   all_empty_idx=all_empty_idx, num_gt=num_gt, hn_ratio=hn_ratio)

        empty_boxes_per_img[b] = sampled_boxes
        selected_indices_per_img[b] = sel_idx

        # cache full sets for redistribution
        empty_boxes_all.append(empty_boxes)
        empty_scores_all.append(empty_scores)
        all_empty_idx_all.append(all_empty_idx)

        # deficit/capacity bookkeeping
        available = empty_boxes.shape[0]
        if available < num_gt:
            deficits.append((b, num_gt, available))
        elif available > num_gt or num_gt == 0:
            capacities.append((b, num_gt, available))

    # Step 3: redistribute deficit across batch
    total_deficit = sum([need - avail for _, need, avail in deficits])


    empty_boxes_per_img = redistribute_deficit(capacities=capacities, total_deficit=total_deficit, empty_boxes_all=empty_boxes_all, 
                                               empty_scores_all=empty_scores_all, all_empty_idx_all=all_empty_idx_all, 
                                               empty_boxes_per_img=empty_boxes_per_img, selected_indices_per_img=selected_indices_per_img, 
                                               hn_ratio=hn_ratio)
    
    # Sanity check
    is_zero_row = (gt_bboxes == 0).all(dim=-1)   
    nonzero_row = ~is_zero_row           
    total_boxes = nonzero_row.sum()  
    total_backgrounds = sum([eb.shape[0] for eb in empty_boxes_per_img])
    diff = total_boxes - total_backgrounds

    if diff / total_boxes > 0.05:
       print(f"Number of objects: {total_boxes} \t Number of background that could be sampled: {total_backgrounds}.")

    return empty_boxes_per_img, diff


def segment2box(segment, width: int = 640, height: int = 640):
    """
    Convert segment coordinates to bounding box coordinates.

    Converts a single segment label to a box label by finding the minimum and maximum x and y coordinates.
    Applies inside-image constraint and clips coordinates when necessary.

    Args:
        segment (torch.Tensor): Segment coordinates in format (N, 2) where N is number of points.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.

    Returns:
        (np.ndarray): Bounding box coordinates in xyxy format [x1, y1, x2, y2].
    """
    x, y = segment.T  # segment xy
    # Clip coordinates if 3 out of 4 sides are outside the image
    if np.array([x.min() < 0, y.min() < 0, x.max() > width, y.max() > height]).sum() >= 3:
        x = x.clip(0, width)
        y = y.clip(0, height)
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """
    Rescale bounding boxes from one image shape to another.

    Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes.
    Supports both xyxy and xywh box formats.

    Args:
        img1_shape (tuple): Shape of the source image (height, width).
        boxes (torch.Tensor): Bounding boxes to rescale in format (N, 4).
        img0_shape (tuple): Shape of the target image (height, width).
        ratio_pad (tuple, optional): Tuple of (ratio, pad) for scaling. If None, calculated from image shapes.
        padding (bool): Whether boxes are based on YOLO-style augmented images with padding.
        xywh (bool): Whether box format is xywh (True) or xyxy (False).

    Returns:
        (torch.Tensor): Rescaled bounding boxes in the same format as input.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def make_divisible(x: int, divisor):
    """
    Return the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def nms_rotated(boxes, scores, threshold: float = 0.45, use_triu: bool = True):
    """
    Perform NMS on oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (torch.Tensor): Rotated bounding boxes with shape (N, 5) in xywhr format.
        scores (torch.Tensor): Confidence scores with shape (N,).
        threshold (float): IoU threshold for NMS.
        use_triu (bool): Whether to use torch.triu operator for upper triangular matrix operations.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
    """
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    if use_triu:
        ious = ious.triu_(diagonal=1)
        # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
        pick = torch.nonzero((ious >= threshold).sum(0) <= 0).squeeze_(-1)
    else:
        n = boxes.shape[0]
        row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
        col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
        upper_mask = row_idx < col_idx
        ious = ious * upper_mask
        # Zeroing these scores ensures the additional indices would not affect the final results
        scores[~((ious >= threshold).sum(0) <= 0)] = 0
        # NOTE: return indices with fixed length to avoid TFLite reshape error
        pick = torch.topk(scores, scores.shape[0]).indices
    return sorted_idx[pick]


def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    in_place: bool = True,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """
    Perform non-maximum suppression (NMS) on prediction results.

    Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. Supports multiple
    detection formats including standard boxes, rotated boxes, and masks.

    Args:
        prediction (torch.Tensor): Predictions with shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing boxes, classes, and optional masks.
        conf_thres (float): Confidence threshold for filtering detections. Valid values are between 0.0 and 1.0.
        iou_thres (float): IoU threshold for NMS filtering. Valid values are between 0.0 and 1.0.
        classes (List[int], optional): List of class indices to consider. If None, all classes are considered.
        agnostic (bool): Whether to perform class-agnostic NMS.
        multi_label (bool): Whether each box can have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A priori labels for each image.
        max_det (int): Maximum number of detections to keep per image.
        nc (int): Number of classes. Indices after this are considered masks.
        max_time_img (float): Maximum time in seconds for processing one image.
        max_nms (int): Maximum number of boxes for torchvision.ops.nms().
        max_wh (int): Maximum box width and height in pixels.
        in_place (bool): Whether to modify the input prediction tensor in place.
        rotated (bool): Whether to handle Oriented Bounding Boxes (OBB).
        end2end (bool): Whether the model is end-to-end and doesn't require NMS.
        return_idxs (bool): Whether to return the indices of kept detections.

    Returns:
        output (List[torch.Tensor]): List of detections per image with shape (num_boxes, 6 + num_masks)
            containing (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        keepi (List[torch.Tensor]): Indices of kept detections if return_idxs=True.
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.stack([torch.arange(len(i), device=prediction.device) for i in xc])[..., None]  # to track idxs

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]  # confidence
        x, xk = x[filt], xk[filt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            xk = xk[filt]

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x, xk = x[filt], xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x, xk = x[filt], xk[filt]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi], keepi[xi] = x[i], xk[i].reshape(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output


def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes (torch.Tensor | np.ndarray): Bounding boxes to clip.
        shape (tuple): Image shape as (height, width).

    Returns:
        (torch.Tensor | np.ndarray): Clipped bounding boxes.
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def clip_coords(coords, shape):
    """
    Clip line coordinates to image boundaries.

    Args:
        coords (torch.Tensor | np.ndarray): Line coordinates to clip.
        shape (tuple): Image shape as (height, width).

    Returns:
        (torch.Tensor | np.ndarray): Clipped coordinates.
    """
    if isinstance(coords, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Rescale masks to original image size.

    Takes resized and padded masks and rescales them back to the original image dimensions, removing any padding
    that was applied during preprocessing.

    Args:
        masks (np.ndarray): Resized and padded masks with shape [H, W, N] or [H, W, 3].
        im0_shape (tuple): Original image shape as (height, width).
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_h, pad_w)).

    Returns:
        (np.ndarray): Rescaled masks with shape [H, W, N] matching original image dimensions.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]

    top, left = (int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1)))
    bottom, right = (
        im1_shape[0] - int(round(pad[1] + 0.1)),
        im1_shape[1] - int(round(pad[0] + 0.1)),
    )

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x, y, width, height) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xywhn2xyxy(x, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | torch.Tensor): Normalized bounding box coordinates in (x, y, w, h) format.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        padw (int): Padding width in pixels.
        padh (int): Padding height in pixels.

    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        clip (bool): Whether to clip boxes to image boundaries.
        eps (float): Minimum value for box width and height.

    Returns:
        (np.ndarray | torch.Tensor): Normalized bounding box coordinates in (x, y, width, height) format.
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xywh2ltwh(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, w, h] where x1, y1 are top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xywh format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def xyxy2ltwh(x):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h] format.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xyxy format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def ltwh2xywh(x):
    """
    Convert bounding boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.

    Args:
        x (torch.Tensor): Input bounding box coordinates.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xywh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    return y


def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation] format.

    Args:
        x (np.ndarray | torch.Tensor): Input box corners with shape (N, 8) in [xy1, xy2, xy3, xy4] format.

    Returns:
        (np.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format with shape (N, 5).
            Rotation values are in radians from 0 to pi/2.
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4] format.

    Args:
        x (np.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format with shape (N, 5) or (B, N, 5).
            Rotation values should be in radians from 0 to pi/2.

    Returns:
        (np.ndarray | torch.Tensor): Converted corner points with shape (N, 4, 2) or (B, N, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """
    Convert bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyxy format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # width
    y[..., 3] = x[..., 3] + x[..., 1]  # height
    return y


def segments2boxes(segments):
    """
    Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh).

    Args:
        segments (list): List of segments where each segment is a list of points, each point is [x, y] coordinates.

    Returns:
        (np.ndarray): Bounding box coordinates in xywh format.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n: int = 1000):
    """
    Resample segments to n points each using linear interpolation.

    Args:
        segments (list): List of (N, 2) arrays where N is the number of points in each segment.
        n (int): Number of points to resample each segment to.

    Returns:
        (list): Resampled segments with n points each.
    """
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments


def crop_mask(masks, boxes):
    """
    Crop masks to bounding box regions.

    Args:
        masks (torch.Tensor): Masks with shape (N, H, W).
        boxes (torch.Tensor): Bounding box coordinates with shape (N, 4) in relative point form.

    Returns:
        (torch.Tensor): Cropped masks.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """
    Apply masks to bounding boxes using mask head output.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).
        upsample (bool): Whether to upsample masks to original image size.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.0)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    Apply masks to bounding boxes using mask head output with native upsampling.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).

    Returns:
        (torch.Tensor): Binary mask tensor with shape (H, W, N).
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def scale_masks(masks, shape, padding: bool = True):
    """
    Rescale segment masks to target shape.

    Args:
        masks (torch.Tensor): Masks with shape (N, C, H, W).
        shape (tuple): Target height and width as (height, width).
        padding (bool): Whether masks are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Rescaled masks.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))) if padding else (0, 0)  # y, x
    bottom, right = (
        mh - int(round(pad[1] + 0.1)),
        mw - int(round(pad[0] + 0.1)),
    )
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize: bool = False, padding: bool = True):
    """
    Rescale segment coordinates from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): Shape of the source image.
        coords (torch.Tensor): Coordinates to scale with shape (N, 2).
        img0_shape (tuple): Shape of the target image.
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_h, pad_w)).
        normalize (bool): Whether to normalize coordinates to range [0, 1].
        padding (bool): Whether coordinates are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def regularize_rboxes(rboxes):
    """
    Regularize rotated bounding boxes to range [0, pi/2].

    Args:
        rboxes (torch.Tensor): Input rotated boxes with shape (N, 5) in xywhr format.

    Returns:
        (torch.Tensor): Regularized rotated boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge if t >= pi/2 while not being symmetrically opposite
    swap = t % math.pi >= math.pi / 2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = t % (math.pi / 2)
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes


def masks2segments(masks, strategy: str = "all"):
    """
    Convert masks to segments using contour detection.

    Args:
        masks (torch.Tensor): Binary masks with shape (batch_size, 160, 160).
        strategy (str): Segmentation strategy, either 'all' or 'largest'.

    Returns:
        (list): List of segment masks as float32 arrays.
    """
    from ultralytics.data.converter import merge_multi_segment

    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":  # merge and concatenate all segments
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """
    Convert a batch of FP32 torch tensors to NumPy uint8 arrays, changing from BCHW to BHWC layout.

    Args:
        batch (torch.Tensor): Input tensor batch with shape (Batch, Channels, Height, Width) and dtype torch.float32.

    Returns:
        (np.ndarray): Output NumPy array batch with shape (Batch, Height, Width, Channels) and dtype uint8.
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def clean_str(s):
    """
    Clean a string by replacing special characters with '_' character.

    Args:
        s (str): A string needing special characters replaced.

    Returns:
        (str): A string with special characters replaced by an underscore _.
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def empty_like(x):
    """Create empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )
