# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys, os, argparse, glob

def update_image(n):
    orb = cv2.ORB_create()
    img = cv2.imread(n)
    keypoint, descriptor = orb.detectAndCompute(img, None)
    descriptor = descriptor.astype(int)
    descriptor = descriptor.tolist()
    return descriptor

