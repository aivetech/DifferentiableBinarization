import tensorflow as tf
import numpy as np
import os.path as osp
import random

mean = [103.939, 116.779, 123.68]


def normalize(img: tf.Tensor, tf_float=tf.float16)->tf.Tensor:
    return tf.image.convert_image_dtype(img, tf_float)


def resize(img: tf.Tensor, width: int, height: int)->tf.Tensor:
    return tf.image.resize(img, [width, height])

def decode_img(file_path: str, width: int, height: int, tf_float=tf.float16)->tf.Tensor:
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = normalize(img)
    img = resize(img, width, height)
    img = tf.cast(img, tf.float32)
    return img

def decode_mask(file_path: str, width: int, height: int)->tf.Tensor:
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = normalize(img)
    img = resize(img, width, height)
    img = tf.reshape(img, [width, height])
    img = tf.cast(img, tf.float32)
    return img

def tf_generator():
    image_size = 640
    is_training = True
    data_dir = 'datasets/total_text_subsample'
    split = 'train' if is_training else 'test'
    with open(osp.join(data_dir, f'{split}_list.txt')) as f:
        image_fnames = f.readlines()
        image_paths = [osp.join(data_dir, f'{split}_images', image_fname.strip()) for image_fname in image_fnames]
        gt_mask_paths = [osp.join(data_dir, f'{split}_gt_masks', image_fname.strip() ) for image_fname in image_fnames]
    
    indice = random.randint(0, 3)
    image = decode_img(image_paths[indice], image_size, image_size)
    image -= mean
    mask = decode_mask(gt_mask_paths[indice], image_size, image_size)
    yield image, mask
