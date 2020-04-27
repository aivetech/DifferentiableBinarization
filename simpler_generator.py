import cv2
import imgaug.augmenters as iaa
import math
import numpy as np
import os.path as osp
import pyclipper
from shapely.geometry import Polygon

from transform import transform, crop, resize

mean = [103.939, 116.779, 123.68]


def load_all_anns(gt_paths, dataset='total_text'):
    res = []
    for gt in gt_paths:
        lines = []
        reader = open(gt, 'r').readlines()
        for line in reader:
            item = {}
            parts = line.strip().split(',')
            label = parts[-1]
            if label == '1':
                label = '###'
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            if 'icdar' == dataset:
                poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
            else:
                num_points = math.floor((len(line) - 1) / 2) * 2
                poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
            if len(poly) < 3:
                continue
            item['poly'] = poly
            item['text'] = label
            lines.append(item)
        res.append(lines)
    return res


def compute_distance(xs, ys, point_1, point_2):
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result


def generate_simpler_model(data_dir, batch_size=16, image_size=640, min_text_size=8, shrink_ratio=0.4, thresh_min=0.3,
             thresh_max=0.7, is_training=True):
    split = 'train' if is_training else 'test'
    with open(osp.join(data_dir, f'{split}_list.txt')) as f:
        image_fnames = f.readlines()
        image_paths = [osp.join(data_dir, f'{split}_images', image_fname.strip()) for image_fname in image_fnames]
        gt_mask_paths = [osp.join(data_dir, f'{split}_gt_masks', image_fname.strip() ) for image_fname in image_fnames]
    transform_aug = iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-10, 10)), iaa.Resize((0.5, 3.0))])
    dataset_size = len(image_paths)
    indices = np.arange(dataset_size)
    if is_training:
        np.random.shuffle(indices)
    current_idx = 0
    b = 0
    while True:
        if current_idx >= dataset_size:
            current_idx = 0
        if b == 0:
            # Init batch arrays
            batch_images = np.zeros([batch_size, image_size, image_size, 3], dtype=np.float32)
            batch_gts = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
        i = indices[current_idx]
        image_path = image_paths[i]
        image = cv2.imread(image_path)
        #if is_training:
        #    transform_aug = transform_aug.to_deterministic()
        #    image, anns = transform(transform_aug, image, anns)
        #    image, anns = crop(image, anns)
        h, w, c = image.shape
        scale_w = image_size / w
        scale_h = image_size / h
        scale = min(scale_w, scale_h)
        h = int(h * scale)
        w = int(w * scale)
        padimg = np.zeros((image_size, image_size, c), image.dtype)
        padimg[:h, :w] = cv2.resize(image, (w, h))
        image = padimg
        gt = np.zeros((image_size, image_size), dtype=np.float32)
        gt = cv2.imread(gt_mask_paths[current_idx], 0)
        padgt = np.zeros((image_size, image_size), image.dtype)
        padgt[:h, :w] = cv2.resize(gt, (w, h))
        gt = np.round(padgt/255)
        
        image = image.astype(np.float32)
        image[..., 0] -= mean[0]
        image[..., 1] -= mean[1]
        image[..., 2] -= mean[2]
        batch_images[b] = image
        
        batch_gts[b] = gt
        
        b += 1
        current_idx += 1
        if b == batch_size:
            inputs = batch_images
            outputs = batch_gts
            yield inputs, outputs
            b = 0


