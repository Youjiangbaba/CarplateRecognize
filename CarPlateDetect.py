#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
	这个文件功能是识别车牌
	1、识别出车牌区域，定位
	2、车牌矫正(仿射变换)、车牌颜色
	3、直方图统计，车牌字符分离
'''
import cv2
import numpy as np
import sys
import os

class Carplatedetect():

	@staticmethod
	#车牌定位	img-输入图片;min_RectArea-最小筛选轮廓面积2000
	def CarPlateLocation(img,min_RectArea):
		cv2.imshow("origin",img)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# 二值化
		ret, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
		#binary = cv2.Canny(binary, 100, 200)#canny算子提取边缘
			# 膨胀和腐蚀操作的核函数
		element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
		element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
			# 膨胀一次，让轮廓突出
		dilation = cv2.dilate(binary, element2, iterations = 1)
			# 腐蚀一次，去掉细节
		erosion = cv2.erode(dilation, element1, iterations = 1)
			# 再次膨胀，让轮廓明显一些
		dilation2 = cv2.dilate(erosion, element2,iterations = 3)
		cv2.imshow("mask_wb",binary)
			####################### 以上为形态学处理，下面进行轮廓筛选，定位车牌#########################
		if 1:
			region = []
				# 查找轮廓
			dilation2,contours,hierarchy = cv2.findContours(dilation2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				# 筛选面积小的
			for i in range(len(contours)):
				cnt = contours[i]
					# 计算该轮廓的面积
				area = cv2.contourArea(cnt)
					# 面积小的都筛选掉
				if (area < min_RectArea):
					continue
					# 轮廓近似，作用很小
				epsilon = 0.001 * cv2.arcLength(cnt,True)
				approx = cv2.approxPolyDP(cnt, epsilon, True)
					# 找到最小的矩形，该矩形可能有方向
				rect = cv2.minAreaRect(cnt)
				print "rect is: "
				print rect

					# box是四个点的坐标
				box = cv2.boxPoints(rect)
				box = np.int0(box)

					# 计算高和宽
				height = abs(box[0][1] - box[2][1])
				width = abs(box[0][0] - box[2][0])
					# 车牌正常情况下长高比在2.7-5之间
				ratio =float(width) / float(height)
				print ratio
				if (ratio > 5 or ratio < 2):
					continue
				print ("get region ok!")
				region.append(box)

			# 用绿线画出这些找到的轮廓
			for boox in region:
				cv2.drawContours(img, [boox], 0, (0, 255, 0), 2)
			cv2.imshow("mask",img)
			ys = [boox[0, 1], boox[1, 1], boox[2, 1], boox[3, 1]]
			xs = [boox[0, 0], boox[1, 0], boox[2, 0], boox[3, 0]]
			ys_sorted_index = np.argsort(ys)
			xs_sorted_index = np.argsort(xs)

			x1 = boox[xs_sorted_index[0], 0]
			x2 = boox[xs_sorted_index[3], 0]

			y1 = boox[ys_sorted_index[0], 1]
			y2 = boox[ys_sorted_index[3], 1]

			img_org2 = img.copy()
			img_plate = img_org2[y1:y2, x1:x2]       
		return dilation2


		
