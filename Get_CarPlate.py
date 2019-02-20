#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
	这个文件功能是识别车牌
	1、分类器得到含有车牌的rect
	2、筛选出正确的车牌区域
	3、定位车牌
	4、直方图统计，车牌字符分离
'''
import cv2
import numpy as np
import sys
import os


