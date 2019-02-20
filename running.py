#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
运行文件
'''
import cv2
import sys
import os
from CarPlateDetect import Carplatedetect

if __name__ == '__main__':
	c = Carplatedetect()
	path = "/home/jiang/图片/car_num/3.jpg"
	image = cv2.imread(path)
	plate = c.CarPlateLocation(image,2000)
	cv2.imshow("oo",plate)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
