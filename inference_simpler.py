import math
import cv2
import os.path as osp
import glob
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import matplotlib.pyplot as plt
from model_simpler import db_simpler
import pytesseract

def resize_image(image, image_short_side=736):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)

    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    points = [[int(p[0]), int(p[1])] for p in points]

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return np.array(box), min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.9):
    pred = pred[..., 0]
    bitmap = bitmap[..., 0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    print(bitmap.shape)
    plt.imshow(bitmap)
    plt.show()
    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #print(contours.shape)

    for contour in contours[:max_candidates]:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        if len(box) == 0:
            continue
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


def new_boxing(pred, bitmap, dest_width, dest_height, max_candidates=1000, box_thresh=0.9):
    pred = pred[..., 0]
    bitmap = bitmap[..., 0]
    height, width = bitmap.shape
    boxes = []
    scores = []
    box_limites, circum_boxes = [], []

    plt.imshow(pred)
    plt.show()
    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #print(contours.shape)

    for contour in contours[:max_candidates]:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=3.0)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        if len(box) == 0:
            continue
        circum_box, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 25:
            continue

        print(box.shape)
        print(circum_box.shape)
        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        circum_box[:, 0] = np.clip(np.round(circum_box[:, 0] / width * dest_width), 0, dest_width)
        circum_box[:, 1] = np.clip(np.round(circum_box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        # x_sorted = sorted(box[:, ], key=lambda k: [k[0], k[1]])
        # y_sorted = sorted(box[:, ], key=lambda k: [k[1], k[0]])
        # x_min = x_sorted[0]
        # x_max = x_sorted[-1]
        # y_min = y_sorted[0]
        # y_max = y_sorted[-1]
        # print(x_min, x_max, y_max, y_min)
        min_y = min([b[0] for b in box])
        max_y = max([b[0] for b in box])
        min_x = min([b[1] for b in box])
        max_x = max([b[1] for b in box])
        box_limites.append([min_x, max_x, min_y, max_y])
        scores.append(score)
        print("found CB")
        print(circum_box)
        circum_boxes.append(circum_box)
    return boxes, scores, box_limites, circum_boxes


if __name__ == '__main__':
    mean = np.array([103.939, 116.779, 123.68])

    _, model = db_simpler()
    model.load_weights('/home/nduforet/Projects/DifferentiableBinarization/checkpoints/2020-02-06/simpler_45_0.0301_0.0297.h5', by_name=True, skip_mismatch=True)
    for image_path in glob.glob(osp.join('/home/nduforet/Projects/CompareST/test_images', 'img195.jpg')):
            image = cv2.imread(image_path)
            src_image = image.copy()
            h, w = image.shape[:2]
            image = resize_image(image)
            image = image.astype(np.float32)
            image -= mean
            plt.imshow(image)
            plt.show()
            image_input = np.expand_dims(image, axis=0)
            p = model.predict(image_input)[0]
            plt.imshow(p[..., 0])
            plt.show()
            bitmap = p > 0.5
            
            #boxes, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=0.001)

            boxes, scores, limites, circum_boxes = new_boxing(p, bitmap, w, h, box_thresh = .01)
            print(scores)

            for box in circum_boxes:
                cv2.drawContours(src_image, [np.array(box)], -1, (0, 255, 0), 2)
            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.imshow('image', src_image)
            #cv2.waitKey(0)
            # for lim in limites:
            #     print(lim)
            #     plt.imshow(src_image[lim[0]:lim[1], lim[2]:lim[3] ])
            #     plt.show()
            #     text = pytesseract.image_to_string(src_image[lim[0]:lim[1], lim[2]:lim[3] ])
            #     print(text)
            image_fname = osp.split(image_path)[-1]
            cv2.imwrite('test/' + image_fname, src_image)
