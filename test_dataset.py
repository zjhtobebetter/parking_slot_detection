import utils.utils_vec
import cv2
import numpy as np

f = open("train.txt")
lines = f.readlines()
for line in lines:
    line_split = line.split()
    img = cv2.imread(line_split[0])
    vec = np.array([np.array(list(map(float, vector.split(',')))) for vector in line_split[1:]])
    if len(vec>0):
        vec = vec[:,-3:]
        utils.utils_vec.show_vec(img,vec)
