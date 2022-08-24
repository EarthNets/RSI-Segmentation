# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import cv2
import numpy as np
import copy

from ..builder import PIPELINES
from .compose import Compose

def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {
        'gt_bboxes': 'gt_labels',
        'gt_bboxes_ignore': 'gt_labels_ignore'
    }
    bbox2mask = {
        'gt_bboxes': 'gt_masks',
        'gt_bboxes_ignore': 'gt_masks_ignore'
    }
    bbox2seg = {
        'gt_bboxes': 'gt_semantic_seg',
    }
    return bbox2label, bbox2mask, bbox2seg



@PIPELINES.register_module()
class RandomRotate90:
    """Apply Rotate Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level should be in range (0,_MAX_LEVEL].
        scale (int | float): Isotropic scale factor. Same in
            ``mmcv.imrotate``.
        center (int | float | tuple[float]): Center point (w, h) of the
            rotation in the source image. If None, the center of the
            image will be used. Same in ``mmcv.imrotate``.
        img_fill_val (int | float | tuple): The fill value for image border.
            If float, the same value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for perform transformation and
            should be in range 0 to 1.
        max_rotate_angle (int | float): The maximum angles for rotate
            transformation.
    """

    def __init__(self,
                 scale=1,
                 center=None,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 prob=1.0):

        assert isinstance(scale, (int, float)), \
            f'The scale must be type int or float. got type {type(scale)}.'
        if isinstance(center, (int, float)):
            center = (center, center)
        elif isinstance(center, tuple):
            assert len(center) == 2, 'center with type tuple must have '\
                f'2 elements. got {len(center)} elements.'
        else:
            assert center is None, 'center must be None or type int, '\
                f'float or tuple, got type {type(center)}.'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, 'img_fill_val as tuple must '\
                f'have 3 elements. got {len(img_fill_val)}.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255]. '\
            f'got {img_fill_val}.'
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
            f'got {prob}.'

        self.scale = scale
        # Rotation angle in degrees. Positive values mean
        # clockwise rotation.
        self.angle = np.random.choice([0, 90, 180, -90])
        self.center = center
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def _rotate_img(self, results, angle, center=None, scale=1.0):
        """Rotate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``mmcv.imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``mmcv.imrotate``.
            scale (int | float): Isotropic scale factor. Same in
                ``mmcv.imrotate``.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_rotated = mmcv.imrotate(
                img, angle, center, scale, border_value=self.img_fill_val)
            results[key] = img_rotated.astype(img.dtype)
            results['img_shape'] = results[key].shape

    def _rotate_bboxes(self, results, rotate_matrix):
        """Rotate the bboxes."""
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = np.stack([[min_x, min_y], [max_x, min_y],
                                    [min_x, max_y],
                                    [max_x, max_y]])  # [4, 2, nb_bbox, 1]
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coordinates = np.concatenate(
                (coordinates,
                 np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype)),
                axis=1)  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.transpose(
                (2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = np.matmul(rotate_matrix,
                                       coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x, min_y = np.min(
                rotated_coords[:, :, 0], axis=1), np.min(
                    rotated_coords[:, :, 1], axis=1)
            max_x, max_y = np.max(
                rotated_coords[:, :, 0], axis=1), np.max(
                    rotated_coords[:, :, 1], axis=1)
            min_x, min_y = np.clip(
                min_x, a_min=0, a_max=w), np.clip(
                    min_y, a_min=0, a_max=h)
            max_x, max_y = np.clip(
                max_x, a_min=min_x, a_max=w), np.clip(
                    max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y],
                                    axis=-1).astype(results[key].dtype)

    def _rotate_masks(self,
                      results,
                      angle,
                      center=None,
                      scale=1.0,
                      fill_val=0):
        """Rotate the masks."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, fill_val)

    def _rotate_seg(self,
                    results,
                    angle,
                    center=None,
                    scale=1.0,
                    fill_val=255):
        """Rotate the segmentation map."""
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            results[key] = mmcv.imrotate(
                seg, angle, center, scale,
                border_value=fill_val).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after rotate
        augmentation."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]

    def __call__(self, results):
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        if np.random.rand() > self.prob:
            return results
        h, w = results['img'].shape[:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        angle = self.angle
        self._rotate_img(results, angle, center, self.scale)
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        self._rotate_bboxes(results, rotate_matrix)
        self._rotate_masks(results, angle, center, self.scale, fill_val=0)
        self._rotate_seg(
            results, angle, center, self.scale, fill_val=self.seg_ignore_label)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'scale={self.scale}, '
        repr_str += f'center={self.center}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob}, '
        return repr_str

