# --------------------------------------------------------
# Faster RCNN
# Image set generation script for corners' data
# Generates training and test set splits
# Written by : Adnan Chaudhry
# --------------------------------------------------------

import os
import os.path as osp
import random
import math
import glob

train_split = 0.75

def get_root_path():
    return osp.abspath(osp.dirname(__file__))

def remove_old_files(files_path):
    files = glob.glob(osp.join(files_path, '*.txt'))
    for file in files:
        os.remove(file)

def get_images(images_path):
    images = []
    project_dirs = next(os.walk(images_path))[1]
    for project_dir in project_dirs:
        image_list = os.listdir(osp.join(images_path, project_dir))
        for image in image_list:
            images.append(project_dir + '/' + image)
    return images

def write_image_set_data(file, data):
    with open(file, 'w') as f:
        for entry in data:
            f.write(entry + '\n')

def generate_splits():
    root_path = get_root_path()
    image_set_path = osp.join(root_path, 'ImageSets')
    remove_old_files(image_set_path)
    images = get_images(osp.join(root_path, 'JPEGImages'))
    num_images = len(images)
    random.seed = 123
    random.shuffle(images)
    train_end_index = int(math.ceil(num_images * train_split))
    train_data = images[:train_end_index]
    test_data = images[train_end_index:]
    write_image_set_data(osp.join(image_set_path, 'trainval.txt'), train_data)
    write_image_set_data(osp.join(image_set_path, 'test.txt'), test_data)

if __name__ == '__main__':
    generate_splits()
