# --------------------------------------------------------
# Faster RCNN
# Data preparation code for corner detection
# Pointivo
# Written by : Adnan Chaudhry
# --------------------------------------------------------

import os
from .imdb import imdb
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
#import cPickle
import subprocess
from ..fast_rcnn.config import cfg

class corners_db(imdb):
    def __init__(self, image_set, corners_db_path=None):
        imdb.__init__(self, 'corners_db_' + image_set)
        self._image_set = image_set
        self._corners_db_path = self._get_default_path() if corners_db_path is None \
                            else corners_db_path
        self._data_path = self._corners_db_path
        self._classes = ('__background__', # always index 0
                         'corner')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.dummy_roidb_handler

        assert os.path.exists(self._corners_db_path), \
                'Corners database path does not exist: {}'.format(self._corners_db_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_image_set_entry(self._image_index[i])

    def image_path_from_image_set_entry(self, image_set_entry):
        """
        Construct an image path from the image's "index" identifier.
        """
        split_entry = image_set_entry.split('/')
        # Get project id
        project_id = split_entry[0]
        # Get image name
        image_name = split_entry[1]
        # Form image path
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  project_id, image_name)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def image_name_at(self, i):
        """
        Return the project ID and name of image at index i in the database
        """
        return self._image_index[i]

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where images and annotations are present
        """
        return os.path.join(cfg.DATA_DIR, 'CornersData')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # Caching disabled for experimentation
        # cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        #     return roidb

        gt_roidb = [self._load_corners_annotation(entry)
                    for entry in self.image_index]
        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def dummy_roidb_handler(self):
        """
        Using RPN with this database so this method is not needed
        """
        pass

    def evaluate_detections(self, all_boxes, output_dir):
        """ Not implementing it for now """
        pass

    def _load_corners_annotation(self, index_entry):
        """
        Load image and bounding boxes info from XML file in the corners annotations
        format.
        """
        split_entry = index_entry.split('/')
        project_id = split_entry[0]
        frame_name_no_ext = os.path.splitext(split_entry[1])[0]
        annotation_filename = project_id + '-' + frame_name_no_ext + '.xml'
        annotation_file = os.path.join(self._data_path, 'Annotations', annotation_filename)
        tree = ET.parse(annotation_file)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}
