import numpy as np
import cv2 as cv
from nltk.metrics import segmentation


def A_LSA_segmentation(Images):
    img = Images
    image_gray = cv.cvtColor(Images, cv.COLOR_BGR2GRAY)
    init_ls = np.zeros(image_gray.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    final_ls = segmentation.chan_vese(img, init_ls)
    return final_ls