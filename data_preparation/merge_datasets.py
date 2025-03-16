import json
import numpy as np
import os
import shutil
import wget
import zipfile
from PIL import Image, ImageOps
import re
import random
from glob import glob

MIDV2020 = "D:\datasets\MIDV2020\photo"
MIDV2020_IMAGE = os.path.join(MIDV2020, 'images')
MIDV2020_ANNOT = os.path.join(MIDV2020, "annotations")

MIDV500 = "D:\datasets\MIDV500"
MIDV500_IMAGE = os.path.join(MIDV500, "images")
MIDV500_ANNOT = os.path.join(MIDV500, "annotations")

DATADIR = "D:\datasets\document_card"
TEMP = os.path.join(DATADIR, "temp")
TEMP_IMAGE = os.path.join(TEMP, "images")
TEMP_ANNOT = os.path.join(TEMP, "annotations")

def read_image(img, annot):
    image = Image.open(img)
    image = ImageOps.exif_transpose(image)
    orig_shape = image.size
    x = np.array(annot['regions'][1]["shape_attributes"]["all_points_x"], dtype=np.int32)
    y = np.array(annot['regions'][1]["shape_attributes"]["all_points_y"], dtype=np.int32)
    coords = np.column_stack((x,y))
    image = image.resize((image.size[0] // 2, image.size[1] // 2))
    normalized_coords = coords.astype(np.float32)
    normalized_coords[:, 0] = normalized_coords[:, 0] / orig_shape[0]
    normalized_coords[:, 1] = normalized_coords[:, 1] / orig_shape[1]
    if not is_valid(normalized_coords):
        return None, None
    annotation = "0 %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (normalized_coords[0, 0], normalized_coords[0, 1], normalized_coords[1, 0], normalized_coords[1, 1], normalized_coords[2, 0], normalized_coords[2, 1], normalized_coords[3, 0], normalized_coords[3, 1])
    return image, annotation

def is_valid(annot: np.ndarray) -> bool:
    return ((annot < 1) & (annot > 0)).all()

def merge_dataset():
    if os.path.exists(DATADIR):
        shutil.rmtree(DATADIR)

    os.mkdir(DATADIR)
    os.mkdir(TEMP)
    os.mkdir(TEMP_IMAGE)
    os.mkdir(TEMP_ANNOT)
    
    file_idx = 0
    for image_folder in os.listdir(MIDV2020_IMAGE):
        image_dir = os.path.join(MIDV2020_IMAGE, image_folder)
        annot_json = json.load(open(os.path.join(MIDV2020_ANNOT, image_folder + ".json")))
        for img, annot in zip(os.listdir(image_dir), annot_json["_via_img_metadata"].values()):
            assert img == annot["filename"]
            image, annotation = read_image(os.path.join(image_dir, img), annot)
            if image is None: continue
            image.save(TEMP_IMAGE + '/image' + str(file_idx) + '.png')
            with open(TEMP_ANNOT + '/image' + str(file_idx) + '.txt', 'w') as f:
                f.write(annotation)
            file_idx += 1

    for img, annot in zip(os.listdir(MIDV500_IMAGE), os.listdir(MIDV500_ANNOT)):
        img_path = os.path.join(MIDV500_IMAGE, img)
        annot_path = os.path.join(MIDV500_ANNOT, annot)
        image = Image.open(img_path)
        image.save(TEMP_IMAGE + '/image' + str(file_idx) + '.png')
        with open(annot_path, 'r') as f:
            annotation = f.readline()
        with open(TEMP_ANNOT + '/image' + str(file_idx) + '.txt', 'w') as f:
            f.write(annotation)
        
        file_idx += 1

def split_train_valid():
    folders = ['train/images', 'train/labels', 'val/images', 'val/labels']

    for folder in folders:
        os.makedirs(os.path.join(DATADIR, folder))

    all_frames = os.listdir(TEMP_IMAGE)
    all_labels = os.listdir(TEMP_ANNOT)

    all_frames.sort(key=lambda var: [int(x) if x.isdigit() else x
                                     for x in re.findall(r'[^0-9]|[0-9]+', var)])
    all_labels.sort(key=lambda var: [int(x) if x.isdigit() else x
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])

    random.seed(230)
    random.shuffle(all_frames)

    # Generate train, val, and test sets for frames
    train_split = int(0.8 * len(all_frames))

    train_frames = all_frames[:train_split]
    val_frames = all_frames[train_split:]

    # Generate corresponding mask lists for masks
    train_labels = [f for f in all_labels if (f[:-4] + '.png') in train_frames]
    val_labels = [f for f in all_labels if (f[:-4] + '.png') in val_frames]

    # Add train, val, test frames and masks to relevant folders
    def add_frames(dir_name, image):
        img = Image.open(os.path.join(TEMP_IMAGE, image))
        img.save(DATADIR + '/{}'.format(dir_name) + '/' + image)

    def add_labels(dir_name, label):
        with open(os.path.join(TEMP_ANNOT, label), 'r') as f:
            data = f.read()

        with open(DATADIR + '/{}'.format(dir_name) + '/' + label, 'w') as f:
            f.write(data)

    frame_folders = [(train_frames, 'train/images'), (val_frames, 'val/images')]
    label_folders = [(train_labels, 'train/labels'), (val_labels, 'val/labels')]

    print('Split images into train, test and validation...')

    # Add frames
    for folder in frame_folders:
        array = folder[0]
        name = [folder[1]] * len(array)
        list(map(add_frames, name, array))

    # Add annots
    for folder in label_folders:
        array = folder[0]
        name = [folder[1]] * len(array)
        list(map(add_labels, name, array))

if __name__ == '__main__':
    # merge_dataset()
    split_train_valid()    
