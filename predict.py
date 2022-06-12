import time

import cv2
import numpy as np
from PIL import Image

from DPSV import DPSV

if __name__ == "__main__":
    dpsv = DPSV()
    mode = "estimate"
    if mode == "predict":
        while True:
            image_path = input("input file path\n")
            try:
                image = Image.open(image_path)
            except:
                print("error in opening img")
                continue
            else:
                angle = 360 * np.random.rand(1)
                image = image.rotate(angle)
                output = dpsv.detect_image(image)
    elif mode == "fps":
        f = open("test_no_repeat.txt")
        annotation_lines = f.readlines()
        start = time.time()
        time1=0
        for i in range(100):
            image_path = annotation_lines[i].split()[0]
            image = Image.open(image_path)
            start = time.time()
            a = dpsv.detect_image(image, show=False)
            end=time.time()
            time1=end-start+time1
        print(1/(time1/100))

    elif mode == "estimate":
        f = open("test_no_repeat.txt")
        f1 = open("error.txt", 'w')
        f2 = open('fail.txt', 'w')
        f3 = open("rad_error.txt", 'w')
        f4 = open("pixel_error.txt", 'w')
        annotation_lines = f.readlines()
        num = np.zeros([1, 3])
        pixel_error = []
        rad_error = []
        b = 0
        for i in range(len(annotation_lines)):
            line = annotation_lines[i]
            num_perimage, pixel_perimage, rad_perimage = dpsv.estimate_per_image(line)
            num = num + num_perimage
            pixel_error = pixel_error + pixel_perimage.tolist()
            rad_error = rad_error + rad_perimage.tolist()
            # if len(pixel_perimage) > 0:
            #     if pixel_perimage.max() > 3:
            #         print(line)
            if num_perimage[0] > num_perimage[2]:
                f2.write(line)
            if num_perimage[1] > num_perimage[2]:
                f1.write(line)
            b = b + 1
            if b % 500 == 0:
                print(b)
        print("slot detection num:", end="\t")
        print(num)
        f3.writelines(str(rad_error))
        f4.writelines(str(pixel_error))
        a = np.array(pixel_error)
        print("pixel error mean and max%s\t%s" % (a.mean(), a.max()))
        a = np.array(rad_error)
        print("rad error mean and max%s\t%s" % (a.mean(), a.max()))

    elif mode == "test":
        # line = "/home/cole/data/DeepPs/ps2.0/testing/outdoor-normal-daylight/463.jpg 1.0,175.99572649572644,554.346153846154,178.70512820512812,411.75641025641033,177.35042735042728,483.0512820512822,0.018999092313133792 1.0,178.70512820512812,411.75641025641033,184.78803418803412,266.6666666666667,181.74658119658113,339.2115384615385,0.04190058910306371 1.0,184.78803418803412,266.6666666666667,189.20683760683755,124.64102564102562,186.99743589743582,195.65384615384616,0.03110268267238392 "
        line = "/home/cole/data/DeepPs/ps2.0/testing/outdoor-normal-daylight/536.jpg 1.0,481.97863247863256,99.79059829059827,469.0128205128207,250.6367521367522,475.4957264957267,175.21367521367523,3.2273357876725974 1.0,469.0128205128207,250.6367521367522,457.85641025641036,400.2905982905984,463.43461538461554,325.4636752136753,3.2160031163493 "
        a = dpsv.estimate_per_image(line)
        print(a)
    elif mode == "fail_read":
        f = open("fail.txt")
        for line in f.readlines():
            print(line)
            line = line.split()
            image = Image.open(line[0])
            output = dpsv.detect_image_no_NMS(image)
    elif mode == "error_read":
        f = open("error.txt")
        for line in f.readlines():
            print(line)
            line = line.split()
            image = Image.open(line[0])
            output = dpsv.detect_image_no_NMS(image)
