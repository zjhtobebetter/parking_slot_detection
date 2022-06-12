import colorsys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from PIL import Image
from nets.yolo import YoloBody
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_vec import decode_outputs, show_vec, generate_center_img, show_vec_two_point


class DPSV(object):
    _defaults = {
        "model_path": 'logs/yolox_s3.pth',
        "classes_path": 'model_data/plot.txt',
        "input_shape": [416, 416],
        "phi": 's',
        "confidence": 0.5,
        "letterbox_image": True,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.class_names, self.num_classes = get_classes(self.classes_path)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self):
        self.net = YoloBody(self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image, show=True):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        outputs = 0
        vec_draw = []
        with torch.no_grad():
            outputs_obj = []
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            # del outputs[0]
            # del outputs[0]
            image = np.array(image_data * 255, dtype=np.uint8).transpose(0, 2, 3, 1)
            image = image[0, :, :, :]
            outputs_decode = decode_outputs(outputs, input_shape=self.input_shape)
            outputs_vec = outputs_decode[..., :4]
            outputs_obj = outputs_decode[..., 4]
            outputs_cls = outputs_decode[..., 5:]
            # max_obj, _ = torch.topk(outputs_obj, 10)
            # choose_obj = max_obj[0, 9]
            # obj_true = outputs_obj > max(0.95, choose_obj)
            obj_true = outputs_obj > 0.99
            cls_true = outputs_cls[obj_true]
            _, cls = torch.topk(cls_true, 1)
            vec_draw = outputs_vec[obj_true].cpu().numpy()
            cls_draw = cls.cpu().numpy()
            if (len(vec_draw) > 0):
                vec_draw = outputs_vec[obj_true]
                obj_draw = outputs_obj[obj_true]
                vec_draw, cls_draw = self.vector_NMS(vectors=vec_draw, objs=obj_draw, cls=cls)
        if (show):
            show_vec_two_point(image, vec_draw, cls_draw)
            cv2.waitKey()
        return vec_draw, cls_draw

    def detect_image_no_NMS(self, image, crop=False):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        outputs = 0
        vec_draw = []
        with torch.no_grad():
            outputs_obj = []
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            # del outputs[0]
            # del outputs[0]
            image = np.array(image_data * 255, dtype=np.uint8).transpose(0, 2, 3, 1)
            image = image[0, :, :, :]
            outputs_decode = decode_outputs(outputs, input_shape=self.input_shape)
            outputs_vec = outputs_decode[..., :4]
            outputs_obj = outputs_decode[..., 4]
            outputs_cls = outputs_decode[..., 5:]
            # max_obj, _ = torch.topk(outputs_obj, 10)
            # choose_obj = max_obj[0, 9]
            # obj_true = outputs_obj > max(0.95, choose_obj)
            obj_true = outputs_obj > 0.99
            cls_true = outputs_cls[obj_true]
            _, cls = torch.topk(cls_true, 1)
            vec_draw = outputs_vec[obj_true].cpu().numpy()
            cls_draw = cls.cpu().numpy()
        show_vec_two_point(image, vec_draw, cls_draw)
        cv2.waitKey()

    def vector_NMS(self, vectors, objs, cls):
        _, max_ind = torch.topk(objs, len(vectors))
        out_vectors = []
        out_cls = []
        out_vectors.append(vectors[max_ind[0], :].cpu().numpy())
        out_cls.append(cls[max_ind[0]].cpu().numpy())
        flag = True
        for i in range(1, len(vectors)):
            for j in range(len(out_vectors)):
                distance = out_vectors[j][0:2] - vectors[max_ind[i], 0:2].cpu().numpy()
                distance = np.sqrt((distance * distance).sum())
                if distance < 75:
                    flag = False
                    break
            if flag:
                out_vectors.append(vectors[max_ind[i], :].cpu().numpy())
                out_cls.append(cls[max_ind[i]].cpu().numpy())
            flag = True
        out_vectors = np.array(out_vectors)
        out_cls = np.array(out_cls)
        return out_vectors, out_cls

    def estimate_per_image(self, annotation_line):
        line = annotation_line.split()
        vec_label = []
        h, w = self.input_shape
        image = Image.open(line[0])
        iw, ih = image.size
        image = cvtColor(image)
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        if len(line) > 1:
            vec_label = np.array([np.array(list(map(float, vec_label.split(',')))) for vec_label in line[1:]])
            vec_label = vec_label[:, -3:]
            np.random.shuffle(vec_label)
            vec_label[:, 0] = vec_label[:, 0] * nw / iw + dx
            vec_label[:, 1] = vec_label[:, 1] * nh / ih + dy
            vec_class = np.array([np.array(list(map(float, vec_label.split(',')))) for vec_label in line[1:]])
            vec_label = np.c_[vec_label, vec_class[:, 0]]
        vec_draw, cls_draw = self.detect_image(image, show=False)

        if len(vec_label) == 0:
            return np.array([0, len(vec_draw), 0]), np.zeros([0]), np.zeros([0])
        label_x = vec_label[:, 0].reshape(-1, 1)
        label_y = vec_label[:, 1].reshape(-1, 1)
        label_rad = np.arctan2(np.sin(vec_label[:, 2]), np.cos(vec_label[:, 2])).reshape(-1, 1)
        predict_x = vec_draw[:, 0].reshape(1, -1)
        predict_y = vec_draw[:, 1].reshape(1, -1)
        predict_rad = np.arctan2(vec_draw[:, 2], vec_draw[:, 3]).reshape(1, -1)
        error_x = (label_x - predict_x) * (label_x - predict_x)
        error_y = (label_y - predict_y) * (label_y - predict_y)
        error_rad = (label_rad - predict_rad) * (label_rad - predict_rad)
        error = error_y + error_x
        matched_vec = error < 800
        match_col = matched_vec.sum(axis=1)
        if match_col.max() > 1:
            for i in range(len(error)):
                if match_col[i] > 1:
                    matched_vec[i, :] = False
                    a = error[i, :].argmin()
                    matched_vec[i, a] = True

        return_pixel_error = np.sqrt(error[matched_vec])
        return_rad_error = np.sqrt(error_rad[matched_vec]).reshape(1, -1)
        return_rad_error1 = (2 * np.pi - return_rad_error).reshape(1, -1)
        return_rad_error=np.row_stack([return_rad_error,return_rad_error1])
        return_rad_error=return_rad_error.min(axis=0)

        return np.array([len(vec_label), len(vec_draw), match_col.sum()]), return_pixel_error, return_rad_error
