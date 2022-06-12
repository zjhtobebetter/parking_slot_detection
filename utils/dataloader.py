from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
# from utils.utils import cvtColor, preprocess_input


#
from utils import cvtColor, preprocess_input
import utils_vec


class DPSV_Dataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mixup, train, mixup_ratio=0.7):
        super(DPSV_Dataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mixup = mixup
        self.train = train
        self.mixup_radio = mixup_ratio
        self.epoch_now = -1
        self.lenth = len(self.annotation_lines)

    def __len__(self):
        return self.lenth

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        # get remainder for item/lenth to avoid the item exceeding the length range
        index = index % self.lenth
        # image, vector = self.get_ori_data(self.annotation_lines[index])
        # image, vector = self.get_ori_data(self.annotation_lines[index])
        lines = sample(self.annotation_lines, 3)
        lines.append(self.annotation_lines[index])
        image, vector = self.get_random_data_Mosaic(annotation_line=lines)
        # image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        vector = np.array(vector, dtype=np.float32)
        vector = self.out_range_process(vector)
        vector = self.vector_to_double_point(vector)
        return image, vector

    def vector_to_double_point(self, vector):
        vec = np.zeros([vector.shape[0], 5])
        if len(vector) > 0:
            vec[..., :2] = vector[..., :2]
            vec[..., 2] = np.sin(vector[..., 2])
            vec[..., 3] = np.cos(vector[..., 2])
            vec[..., 4] = vector[..., 3]
        return vec

    def out_range_process(self, vectors):
        in_range_vector = []
        for i in range(len(vectors)):
            if vectors[i, 0] >= 0 and vectors[i, 0] < self.input_shape[0] and vectors[i, 1] >= 0 and vectors[i, 1] < \
                    self.input_shape[1]:
                if vectors[i, 2] > np.pi:
                    vectors[i, 2] = vectors[i, 2] - 2 * np.pi
                if vectors[i, 2] < -np.pi:
                    vectors[i, 2] = vectors[i, 2] + 2 * np.pi
                in_range_vector.append(vectors[i])
        return np.array(in_range_vector)

    def get_ori_data(self, annotation_line, input_shape=[416, 416]):
        line = annotation_line.split()
        h, w = input_shape
        image = Image.open(line[0])
        iw, ih = image.size
        image = cvtColor(image)
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        vector = []
        if len(line) > 1:
            vector = np.array([np.array(list(map(float, vector.split(',')))) for vector in line[1:]])
            vector = vector[:, -3:]
            np.random.shuffle(vector)
            vector[:, 0] = vector[:, 0] * nw / iw + dx
            vector[:, 1] = vector[:, 1] * nh / ih + dy
        if len(line) > 1:
            vec_class = np.array([np.array(list(map(float, vector.split(',')))) for vector in line[1:]])
            vector = np.c_[vector, vec_class[:, 0]]
        image_data = np.array(new_image, np.uint8)
        return image_data, vector

    def get_random_data(self, annotation_line, input_shape=[416, 416]):
        line = annotation_line.split()
        h, w = input_shape
        image = Image.open(line[0])
        iw, ih = image.size
        image = cvtColor(image)
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        vector = []
        if len(line) > 1:
            vector = np.array([np.array(list(map(float, vector.split(',')))) for vector in line[1:]])
            vector = vector[:, -3:]
            np.random.shuffle(vector)
            vector[:, 0] = vector[:, 0] * nw / iw + dx
            vector[:, 1] = vector[:, 1] * nh / ih + dy
        # 图像是否翻转
        flip = self.rand() < 0.5
        if True:
            new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(vector) > 0:
                vector = self.vector_flip(vector, image.size)
        # 图片是否旋转
        rotate = self.rand() < 0.8
        if True:
            angle = self.rand() * 360
            new_image = new_image.rotate(angle)
            if len(line) > 1:
                points = []
                points_ = np.array([np.array(list(map(float, vector.split(',')))) for vector in line[1:]])
                points_ = points_[:, 1:5]
                points_[:, 0] = points_[:, 0] * nw / iw + dx
                points_[:, 1] = points_[:, 1] * nh / ih + dy
                points_[:, 2] = points_[:, 2] * nw / iw + dx
                points_[:, 3] = points_[:, 3] * nh / ih + dy
                for i in range(len(points_)):
                    point = np.array([points_[i, 0], points_[i, 1]])
                    points.append(point)
                    point = np.array([points_[i, 2], points_[i, 3]])
                    points.append(point)
                points = np.array(points)
                points = self.get_rotated_point(points, angle_rad=angle / 360 * 2 * np.pi, input_shape=input_shape)
                vector[:, 0:2] = self.get_rotated_point(vector[:, 0:2], angle_rad=angle / 360 * 2 * np.pi,
                                                        input_shape=input_shape)
                if len(line) > 1:
                    vec_class = np.array([np.array(list(map(float, vector.split(',')))) for vector in line[1:]])
                    vector = np.c_[vector, vec_class[:, 0]]
                new_vector = []
                for i in range(len(vector)):
                    j = 2 * i
                    if points[j:j + 2, 0].max() < input_shape[0] and points[j:j + 2, 0].min() >= 0 and points[j:j + 2,
                                                                                                       1].max() < \
                            input_shape[1] and points[j:j + 2, 1].min() >= 0:
                        vector[i, 2] = vector[i, 2] - angle / 360 * 2 * np.pi
                        new_vector.append(vector[i])
                new_vector = np.array(new_vector)
                vector = new_vector
        image_data = np.array(new_image, np.uint8)
        return image_data, vector

    def get_rotated_point(self, points, angle_rad, input_shape):
        cx = input_shape[0] / 2
        cy = input_shape[1] / 2
        new_points = np.zeros_like(points)
        new_points[:, 0] = (points[:, 0] - cx) * np.cos(angle_rad) + (points[:, 1] - cy) * np.sin(angle_rad) + cx
        new_points[:, 1] = (points[:, 1] - cy) * np.cos(angle_rad) - (points[:, 0] - cx) * np.sin(angle_rad) + cy
        return new_points

    def vector_flip(self, vector, image_size):
        w, h = image_size
        vector[:, 0] = w - vector[:, 0]
        for i in range(len(vector)):
            if vector[i, 2] >= 0:
                vector[i, 2] = -vector[i, 2] + np.pi
            else:
                vector[i, 2] = -vector[i, 2] - np.pi
        return vector

    def get_random_data_Mosaic(self, annotation_line, input_shape=[416, 416]):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # ---------------------------------#
            #   每一行进行分割
            # ---------------------------------#
            line_content = line.split()
            # ---------------------------------#
            #   打开图片
            # ---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            vector = []
            if len(line_content) > 1:
                vector = np.array([np.array(list(map(float, vector.split(',')))) for vector in line_content[1:]])
                vector = vector[:, -3:]
                np.random.shuffle(vector)
                vector[:, 0] = vector[:, 0] * nw / iw + dx
                vector[:, 1] = vector[:, 1] * nh / ih + dy
            # 图像是否翻转
            flip = self.rand() < 0.5
            if True:
                new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
                if len(vector) > 0:
                    vector = self.vector_flip(vector, image.size)

            angle = self.rand() * 360
            angle = 0
            new_image = new_image.rotate(angle)
            # -----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            # -----------------------------------------------#
            if index == 0:
                dx = int(nw * min_offset_x) - nw
                dy = int(nh * min_offset_y) - nh
            elif index == 1:
                dx = int(nw * min_offset_x) - nw
                dy = int(nh * min_offset_y)
            elif index == 2:
                dx = int(nw * min_offset_x)
                dy = int(nh * min_offset_y)
            elif index == 3:
                dx = int(nw * min_offset_x)
                dy = int(nh * min_offset_y) - nh
            new_image.paste(new_image, (dx, dy))
            image_data = np.array(new_image, np.uint8)
            index = index + 1
            vec_datas = []
            if len(line_content) > 1:
                points = []
                points_ = np.array([np.array(list(map(float, vector.split(',')))) for vector in line_content[1:]])
                points_ = points_[:, 1:5]
                points_[:, 0] = points_[:, 0] * nw / iw + dx
                points_[:, 1] = points_[:, 1] * nh / ih + dy
                points_[:, 2] = points_[:, 2] * nw / iw + dx
                points_[:, 3] = points_[:, 3] * nh / ih + dy
                for i in range(len(points_)):
                    point = np.array([points_[i, 0], points_[i, 1]])
                    points.append(point)
                    point = np.array([points_[i, 2], points_[i, 3]])
                    points.append(point)
                points = np.array(points)
                points = self.get_rotated_point(points, angle_rad=angle / 360 * 2 * np.pi, input_shape=input_shape)
                points[:, 0] = points[:, 0] + dx
                points[:, 1] = points[:, 1] + dy
                vector[:, 0:2] = self.get_rotated_point(vector[:, 0:2], angle_rad=angle / 360 * 2 * np.pi,
                                                        input_shape=input_shape)
                vector[:, 0] = vector[:, 0] + dx
                vector[:, 1] = vector[:, 1] + dy
                print(points)
                for i in range(len(vector)):
                    j = 2 * i
                    if points[j:j + 2, 0].max() < input_shape[0] and points[j:j + 2, 0].min() >= 0 and points[
                                                                                                       j:j + 2,
                                                                                                       1].max() < \
                            input_shape[1] and points[j:j + 2, 1].min() >= 0:
                        vector[i, 2] = vector[i, 2] - angle / 360 * 2 * np.pi
                        vec_datas.append(vector[i])

            image_datas.append(image_data)
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        vector = np.array(vec_datas)
        return new_image, vector

    def get_new_theta(self, theta, iw, ih, nw, nh):
        theta_x = np.cos(theta)
        theta_y = np.sin(theta)
        new_theta_x = theta_x * nw / iw
        new_theta_y = theta_y * nh / ih
        new_theta = np.arctan2(new_theta_y, new_theta_x)
        return new_theta


def DPSV_dataset_collate(batch):
    images = []
    vectors = []
    for img, vector in batch:
        images.append(img)
        vectors.append(vector)
    images = np.array(images)
    return images, vectors


if __name__ == "__main__":

    f = open("/home/cole/code/python/parking-slot-detection_with_two_points/test_right.txt")
    annotation_lines = f.readlines()
    test_dataset = DPSV_Dataset(annotation_lines, [416, 416], 3, 100, True, True, 0.7)
    # img, vec = test_dataset[439]
    # print(vec)
    # utils_vec.show_vec(img, vec)
    for i in range(2679, 2726):
        img, vec = test_dataset[i]
        print(img.shape)
        utils_vec.show_vec_two_point(img, vec)
        cv2.waitKey()
