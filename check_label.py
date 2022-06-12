from PIL import Image
import numpy as np
import cv2
from utils.utils import cvtColor,preprocess_input
line=input("input img file\n")
image=cv2.imread(line)
cv2.imshow("a",image)
cv2.waitKey()