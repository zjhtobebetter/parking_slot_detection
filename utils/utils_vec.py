import cv2
import numpy as np
import torch


def show_vec(img, vecs,name="arrow"):
    for vector in vecs:
        point1 = (round(vector[0]), round(vector[1]))
        point2 = (round(point1[0] - 50 * np.cos(vector[2])), round(point1[1] - 50 * np.sin(vector[2])))
        cv2.arrowedLine(img, point1, point2, [0, 255, 255])
    cv2.imshow(name, img)
    # cv2.waitKey()

def show_vec_two_point(img,vecs,cls=np.array([[1,1]]),name="arrow"):
    for vector in vecs:
        color = [0, 0, 0]
        color[round(cls[0,0])-1]=255
        point1 = (round(vector[0]), round(vector[1]))
        point2 = (round(point1[0] + 50 * vector[3]), round(point1[1] + 50 * vector[2]))
        cv2.arrowedLine(img, point1, point2, color)
    cv2.imshow(name, img)

def generate_center_img(outputs, input_shape):
    hw = [x.shape[-2:] for x in outputs]
    grids = []
    strides = []
    centers = []
    for h, w in hw:
        stride_x = input_shape[0] / w
        stride_y = input_shape[1] / h
        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        center = grid
        center[..., 0] = center[..., 0] * stride_x
        center[..., 1] = center[..., 1] * stride_y
        centers.append(center)
    return centers


def decode_outputs(outputs, input_shape):
    grids = []
    strides = []
    hw = [x.shape[-2:] for x in outputs]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)

    for h, w in hw:
        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    return outputs
