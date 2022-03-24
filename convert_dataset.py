import argparse
import glob
import json
import math
import os
from os.path import join, basename, exists

import cv2
import numpy as np


def read_json(filename):
    data = None
    if exists(filename):
        fp = open(filename)
        data = json.load(fp)

    return data


def append_image(filename, points, validation=0.0):
    pts = list()
    bbox = cv2.boundingRect(np.array(points))
    bbox = [[bbox[0] - 5, bbox[1] - 5], [bbox[0] + bbox[2] +
                                         10, bbox[1] + bbox[3] + 10]]  # 5 pixles por cada lado
    item = dict()
    item['dataset'] = 'custom'
    item['isValidation'] = validation
    item['img_paths'] = filename
    item['objpos'] = points[0]

    for idx in range(1,len(points)):
        pt = points[idx]
        pts.append([ pt[0], pt[1], 2.0 ])
    item['joint_self'] = pts
    item['scale_provided'] = 0.72

    return item


def get_base_video_names(image_path):
    files = glob.glob(join(image_path, "*_001.jpg"))

    video_names = list()

    for file in files:
        filename = basename(file)
        base = filename.replace(filename.split("_")[-1], '')
        video_names.append(base)

    return video_names


def draw_image(img, json):
    points = json['points']
    bbox = json['bbox']
    img_copy = img.copy()

    pt1 = (bbox[0][0], bbox[0][1])
    pt2 = (bbox[1][0], bbox[1][1])

    for pt in points:
        cv2.circle(img=img_copy, center=(
            pt[0], pt[1]), radius=5, color=(0, 0, 255), thickness=-1)

    cv2.rectangle(img=img_copy, pt1=pt1, pt2=pt2, color=(255, 0, 0))

    return img_copy


def main(args):
    input_images = glob.glob(join(args.input_path, "inputs", "*.jpg"))

    output_image_path = join(args.output_path, "images")
    output_annotated_image_path = join(args.output_path, "annotated_images")
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_annotated_image_path, exist_ok=True)

    train_json_filename = join(args.output_path, "annotations.json")

    videonames = get_base_video_names(join(args.input_path, "inputs"))

    test_videos = math.floor(len(videonames) * args.test)
    if test_videos < 1:
        test_videos = 1

    val_videos = math.floor(len(videonames) * args.val)
    if val_videos < 1:
        val_videos = 1

    train_videos = len(videonames) - test_videos - val_videos

    list_train_videos = videonames[:train_videos]
    list_test_videos = videonames[train_videos:train_videos+test_videos]
    list_val_videos = videonames[train_videos+test_videos:]

    train_annotations = list()

    for idx, image_path in enumerate(input_images):
        filename = basename(image_path)
        basefilename = filename.replace(filename.rsplit("_")[-1], '')
        jsonfilename = filename.replace('.jpg', '.json')
        json_path = join(args.input_path, "labels", jsonfilename)

        if exists(image_path) and exists(json_path):
            points = []
            data = read_json(json_path)
            img = cv2.imread(image_path)
            fx = img.shape[0] / 48.0
            fy = img.shape[1] / 48.0
            for idx2, item in enumerate(data[0]['center']):
                x = math.ceil(item['coordinates']['x'] * fx)
                y = math.ceil(item['coordinates']['y'] * fy)
                points.append([x, y])
                #img = draw_point(img, x, y, str(idx2))

            # Copy image to output
            cv2.imwrite(join(output_image_path, filename), img)

            #shutil.copy(image_path, join(output_image_path, filename))
            validation = 0.0

            if basefilename in list_val_videos:
                validation = 1.0
            image_json = append_image(filename, points, validation)

            train_annotations.append(image_json)

            #anno_img = draw_image(img, image_json)
            #cv2.imwrite(join(output_annotated_image_path, filename), anno_img)

            

    train_fp = open(train_json_filename, "w")
    json.dump(train_annotations, train_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input path')
    parser.add_argument('output_path', type=str, help='Output path')
    parser.add_argument('--train', type=float, default=0.8)
    parser.add_argument('--val', type=float, default=0.1)
    parser.add_argument('--test', type=float, default=0.1)
    args = parser.parse_args()

    main(args)
