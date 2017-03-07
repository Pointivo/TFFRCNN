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
#import cPickle
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
        # Default roidb handler
        self._roidb_handler = self.gt_roidb

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

    def compute_det_gt_overlaps(self, detected_boxes, gt_box, overlap_thresh=0.16, conf_thresh=0.05):
        """
        Returns which detected boxes overlap with the provided ground truth box
        """
        if (gt_box.size == 0) or (detected_boxes.size == 0):
            return np.array([], dtype=np.uint32), np.array([], dtype=np.float)
        # Compute intersections
        ixmin = np.maximum(detected_boxes[:, 0], gt_box[0])
        iymin = np.maximum(detected_boxes[:, 1], gt_box[1])
        ixmax = np.minimum(detected_boxes[:, 2], gt_box[2])
        iymax = np.minimum(detected_boxes[:, 3], gt_box[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        intersection_areas = iw * ih
        # mask out intersections with detections having confidence values lesser than a threshold
        mask_intersections = np.where(detected_boxes[:, 4] < conf_thresh)[0]
        intersection_areas[mask_intersections] = 0.
        # Compute GT box area
        gt_area = (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.)
        # Compute unions
        union_areas = (gt_area + (detected_boxes[:, 2] - detected_boxes[:, 0] + 1.) *
                      (detected_boxes[:, 3] - detected_boxes[:, 1] + 1.) - intersection_areas)
        # Intersection over union area overlap ratios
        IoU_overlap_ratios = intersection_areas / union_areas
        # Intersection over ground truth area overlap ratios
        I_overlap_ratios = intersection_areas / gt_area
        return np.where(IoU_overlap_ratios > overlap_thresh)[0], I_overlap_ratios

    def getRelevanceMeasures(self, detected_boxes, gt_boxes):
        """
         Return the number of true positives, false positives and false negatives
         This method assumes that the detected_boxes and gt_boxes belong to the same class
        """
        num_det_boxes = len(detected_boxes)
        num_gt_boxes = len(gt_boxes)
        tp_map = np.zeros(num_det_boxes, dtype=np.uint8)
        fn_map = np.ones(num_gt_boxes, dtype=np.uint8)
        I_overlaps = np.array([], dtype=np.float)
        for gt_ind in xrange(num_gt_boxes):
            overlap_inds, intersection_overlap_ratios = self.compute_det_gt_overlaps(detected_boxes, gt_boxes[gt_ind, :])
            I_overlaps = np.concatenate((I_overlaps, intersection_overlap_ratios), axis=0)
            if overlap_inds.size != 0:
                tp_map[overlap_inds] = 1
                fn_map[gt_ind] = 0
        num_tp = np.count_nonzero(tp_map)
        num_fp = num_det_boxes - num_tp
        num_fn = np.count_nonzero(fn_map)
        return num_tp, num_fp, num_fn, I_overlaps

    def evaluate_detections(self, all_boxes, output_dir=None):
        """ Output precision and recall measures for the detections """
        if output_dir is None:
            output_dir = '.'
        num_images = len(self.image_index)
        tps = 0
        fps = 0
        fns = 0
        all_intersection_overlaps = np.array([], dtype=np.float)
        for image_ind in xrange(num_images):
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                gt_boxes_inds_cls = np.where(self.roidb[image_ind]['gt_classes'] == cls_ind)[0]
                gt_boxes_cls = self.roidb[image_ind]['boxes'][gt_boxes_inds_cls]
                cls_tp_per_im, cls_fp_per_im,\
                cls_fn_per_im, cls_im_intersection_overlaps = self.getRelevanceMeasures(all_boxes[cls_ind][image_ind],
                                                                                        gt_boxes_cls)
                tps += cls_tp_per_im
                fps += cls_fp_per_im
                fns += cls_fn_per_im
                all_intersection_overlaps = np.concatenate((all_intersection_overlaps, cls_im_intersection_overlaps),
                                                           axis=0)
        # cfg.EPS is a small number which avoids division by zero
        precision = tps / (tps + fps + cfg.EPS)
        recall = tps / (tps + fns + cfg.EPS)
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Evaluation done over a set of {:d} test images'.format(num_images))
        print('Average Precision = {:.4f}'.format(precision))
        print('Average Recall = {:.4f}'.format(recall))
        print('--------------------------------------------------------------')
        import matplotlib.pyplot as plt
        fig, _ = plt.subplots(figsize=(12, 12))
        fig.clear()
        all_intersection_overlaps = all_intersection_overlaps[np.where(all_intersection_overlaps > 0.)[0]]
        hist, bin_edges = np.histogram(all_intersection_overlaps, bins=10, normed=True)
        bin_width = bin_edges[1] - bin_edges[0]
        plt.bar(bin_edges[:-1], hist * bin_width, bin_width)
        plt.title('Histogram of detection-ground truth overlap ratios over a test set of {:d} images'.format(num_images))
        plt.xlabel('Ratio of overlap area to ground truth area')
        plt.ylabel('Proportion of detected boxes')
        plt.savefig(os.path.join(output_dir, 'overlap_hist.jpg'))

    def _load_corners_annotation(self, index_entry):
        """
        Load image and bounding boxes info from XML file in the corners annotations
        format.
        """
        split_entry = index_entry.split('/')
        project_id = split_entry[0]
        frame_name_no_ext = os.path.splitext(split_entry[1])[0]
        annotation_filename = frame_name_no_ext + '.xml'
        annotation_file = os.path.join(self._data_path, 'Annotations', project_id, annotation_filename)
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
