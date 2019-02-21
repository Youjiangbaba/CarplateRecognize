#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
	这个文件功能是识别车牌
	1、分类器得到含有车牌的rect
	2、筛选出正确的车牌区域
	3、定位车牌
	4、直方图统计，车牌字符分离
'''
import numpy as np
import cv2
import os
import time
import numpy as np
import copy
import sys
import copy
from matplotlib import pyplot as plt

reload(sys)
sys.setdefaultencoding('utf8')

green_min = np.array([35, 100, 46]) 
green_max = np.array([70, 255, 255])

yello_min = np.array([20, 100, 46]) 
yello_max = np.array([30, 255, 255])

blue_min = np.array([100, 100, 46]) 
blue_max = np.array([124, 255, 255]) 

white_min = np.array([40, 40, 100]) 
white_max = np.array([150, 200, 255]) 
#读取图片文件
def imreadex(filename):
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
	
def point_limit(point):
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0

#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
	up_point = -1#上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []

	for i,x in enumerate(histogram):
		if is_peak and x < threshold:#如果当前状态是波峰，且当前位置的值小于阈值
			if i - up_point > 2:#但前位置与上升点的距离大于2
				is_peak = False#波峰到此位置
				wave_peaks.append((up_point, i))#记录下这个波峰的起始位置和终止位置
		elif not is_peak and x >= threshold:#如果当前状态不是波峰，且当前位置的值大于阈值
			is_peak = True#将当前状态修改为：处于波峰
			up_point = i#记录上升点位置
	#记录下最后一个波峰
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks

#根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
	part_cards = []
	for wave in waves:
		part_cards.append(img[:, wave[0]:wave[1]])
	return part_cards

#精确定位，返回颜色（b y g）--------------------------------------------------------------------------------------------------------------------------------------------------------
def GetCarplate(img,watches):
	extend_scale = 1;
	shape = [140,180]
	color = 0
	for (x, y, w, h) in watches:
		#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
		patch  = img[y:y + h, x:x + w]
		#转为hsv 求h通道，得出蓝色阈值，取该阈值二值化
		pat_hsv = img[y:y + h, x:x + w]
		patch_hsv = cv2.cvtColor(pat_hsv, cv2.COLOR_BGR2HSV)
		mask_b=cv2.inRange(patch_hsv,  blue_min,  blue_max)

		mask_g=cv2.inRange(patch_hsv,  green_min,  green_max)

		mask_y=cv2.inRange(patch_hsv,  yello_min,  yello_max)

		sumb_mask = mask_b.sum()
		sumg_mask = mask_g.sum()
		sumy_mask = mask_y.sum()
		#cv2.imshow("maskb",mask_b)
		#cv2.imshow("maskg",mask_g)
		#cv2.imshow("masky",mask_y)
		print ("sum: ",sumb_mask,sumg_mask,sumy_mask)
		if sumb_mask > 10000: 				#蓝色干扰,过滤非车牌			
			#得到蓝色车牌大致位置，扩大区域，形态学处理精确定位
			#cv2.imshow("m"+str(count),mask_b)
			color = 'b'
			extend_w = int(w*extend_scale)
			extend_h = int(h*extend_scale)
			start_corr_y = y - int(extend_h*0.3)
			end_coor_y  =y + extend_h + int(extend_h*0.3)
			start_corr_x = x - int(extend_w*0.3)
			end_coor_x = x + extend_w + int(extend_w*0.3)

			if start_corr_x < 0 :
				start_corr_x = 0;
			if start_corr_y < 0 :
				start_corr_y = 0;
			if end_coor_x>img.shape[1]:
				end_coor_x = img.shape[1]
			if end_coor_y>img.shape[0]:
				end_coor_y = img.shape[0]
			extend_patch = img[start_corr_y:end_coor_y, start_corr_x:end_coor_x]
			cv2.imshow("0",extend_patch)
####


###
			#根据颜色边界定位
			hsv_mid = cv2.cvtColor(extend_patch, cv2.COLOR_BGR2HSV)
			row_num, col_num = hsv_mid.shape[:2]
			xl = col_num
			xr = 0
			yb = 0
			yt = row_num
			#根据颜色确定车牌图像的上下边界
			for i in range(row_num):
				count = 0
				for j in range(col_num):
					H = hsv_mid.item(i, j, 0)
					S = hsv_mid.item(i, j, 1)
					V = hsv_mid.item(i, j, 2)
					B = extend_patch.item(i,j,0)
					G = extend_patch.item(i,j,1)
					R = extend_patch.item(i,j,2)
					if blue_min[0] < H <= blue_max[0] and 34 < S and 46 < V and B-5 > G and B-15 > R and B > 85 and R < 120 and G <150:
						count += 1

				if count > 50:
					if yt > i:
						yt = i
					if yb < i:
						yb = i

			#根据颜色确定车牌图像的左右边界
			for j in range(col_num):
				count = 0
				for i in range(row_num):
					H = hsv_mid.item(i, j, 0)
					S = hsv_mid.item(i, j, 1)
					V = hsv_mid.item(i, j, 2)
					B = extend_patch.item(i,j,0)
					G = extend_patch.item(i,j,1)
					R = extend_patch.item(i,j,2)
					if blue_min[0] < H <= blue_max[0] and 34 < S and 46 < V and B-5 > G and B-15 > R and B > 85 and R < 120 and G <150:
						count += 1

				if count > 20:
					if xl > j:
						xl = j
					if xr < j:
						xr = j
			extend_patch = extend_patch[yt:yb,xl:xr]
			return mask_b,extend_patch,color                      #定位完成

		elif sumg_mask > 10000: 						
			#得到蓝色车牌大致位置，扩大区域，形态学处理精确定位
			#cv2.imshow("m"+str(count),mask_b)
			color = 'g'
			extend_w = int(w*extend_scale)
			extend_h = int(h*extend_scale)
			start_corr_y = y - int(extend_h*0.1)
			end_coor_y  =y + extend_h + int(extend_h*0.1)
			start_corr_x = x - int(extend_w*0.1)
			end_coor_x = x + extend_w + int(extend_w*0.1)

			if start_corr_x < 0 :
				start_corr_x = 0;
			if start_corr_y < 0 :
				start_corr_y = 0;
			if end_coor_x>img.shape[1]:
				end_coor_x = img.shape[1]
			if end_coor_y>img.shape[0]:
				end_coor_y = img.shape[0]
			extend_patch = img[start_corr_y:end_coor_y, start_corr_x:end_coor_x]

			return mask_g,extend_patch,color                      #定位完成

		elif sumy_mask > 10000: 				#蓝色干扰,过滤非车牌			
			#得到蓝色车牌大致位置，扩大区域，形态学处理精确定位
			#cv2.imshow("m"+str(count),mask_b)
			color = 'y'
			extend_w = int(w*extend_scale)
			extend_h = int(h*extend_scale)
			start_corr_y = y - int(extend_h*0.2)
			end_coor_y  =y + extend_h + int(extend_h*0.2)
			start_corr_x = x - int(extend_w*0.2)
			end_coor_x = x + extend_w + int(extend_w*0.2)

			if start_corr_x < 0 :
				start_corr_x = 0;
			if start_corr_y < 0 :
				start_corr_y = 0;
			if end_coor_x>img.shape[1]:
				end_coor_x = img.shape[1]
			if end_coor_y>img.shape[0]:
				end_coor_y = img.shape[0]
			extend_patch = img[start_corr_y:end_coor_y, start_corr_x:end_coor_x]
			return mask_y,extend_patch,color                      #定位完成
	return img,img,0


#形态学处理，返回明显字符的二值图像 ---------------------------------------------------------------------------------------------------------------------------------
def morphology_deal(img):

	# 再次膨胀，让轮廓明显一些
	#dilation2 = cv2.dilate(dilation, element2,iterations = 1)
	if 1:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)       #大津阈值法
		#开运算，消除噪声
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))         #定义矩形结构元素
		closed1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,iterations=1)    #闭运算1
		erosion = cv2.morphologyEx(closed1, cv2.MORPH_OPEN, kernel,iterations=1)
		closed1 = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel,iterations=1)    #闭运算2
		# 腐蚀一次，去掉细节
		#erosion = cv2.erode(binary, np.uint8(np.zeros((7,7))))
		#erosion = cv2.erode(erosion, np.uint8(np.zeros((7,7))))
	#binary = cv2.inRange(img, np.array([60, 60, 60]),np.array([255, 255, 255]))
	#binary=cv2.bitwise_not(binary)
	return closed1

#直方图，字符分离 --------------------------------------------------------------------------------------------------------------------------------------------------------
def Character_separation(gray_img):
	#查找竖直直方图波峰
	#参数axis为1表示压缩列，将每一行的元素相加，将矩阵压缩为一列
	x_histogram  = np.sum(gray_img, axis=1)
	#plt.figure('竖直方向波峰图')
	#plt.plot(x_histogram)
	# plt.show()
	x_min = np.min(x_histogram)
	x_average = np.sum(x_histogram)/x_histogram.shape[0]
	x_threshold = (x_min + x_average)/2

	#使用自定义的函数寻找波峰，从而分割字符
	wave_peaks = find_waves(x_threshold, x_histogram)
	if len(wave_peaks) == 0:
		print("peak less 0:")

	#认为竖直方向，最大的波峰为车牌区域
	wave = max(wave_peaks, key=lambda x:x[1]-x[0])
	gray_img = gray_img[wave[0]:wave[1]]
	#plt.figure('字符所在的区域的车牌图像')
	#plt.imshow(gray_img, cmap = 'gray')
								
	#查找水平直方图波峰
	row_num, col_num= gray_img.shape[:2]
	#去掉车牌上下边缘1个像素，避免白边影响阈值判断
	gray_img = gray_img[1:row_num-1]
	#参数axis为0表示压缩列，将每一列的元素相加，将矩阵压缩为一行
	y_histogram = np.sum(gray_img, axis=0)

	#plt.figure('水平方向波峰图')
	#plt.plot(y_histogram)
	#plt.show()	
	
	y_min = np.min(y_histogram)
	y_average = np.sum(y_histogram)/y_histogram.shape[0]
	y_threshold = (y_min + y_average)/10					#U和0要求阈值偏小，否则U和0会被分成两半

	wave_peaks = find_waves(y_threshold, y_histogram)
	#for wave in wave_peaks:
	#	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2) 
	#车牌字符数应大于6
	if len(wave_peaks) <= 6:
		print("peak less 1:", len(wave_peaks))
				
	#找出宽度最大的波峰
	wave = max(wave_peaks, key=lambda x:x[1]-x[0])
	max_wave_dis = wave[1] - wave[0]
				
	#判断是否是左侧车牌边缘
	if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
		#如果是左侧车牌边缘，则将其剔除
		wave_peaks.pop(0)
	#####################################################寻找直方图中的波峰（结束）##########################################

	########################################组合汉字，去除车牌上的分割点（开始）#############################################
	#组合汉字（一个汉字可能由好几个连续波峰组成）
	#一个汉字可能由好几个波峰组成，找到这几个波峰，并且将它们合并在一起
	cur_dis = 0
	for i, wave in enumerate(wave_peaks):
		if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
			break
		else:
			cur_dis += wave[1] - wave[0]
	if i > 0:
		#这种情况说明，前几个波峰的组合代表一个汉字
		wave = (wave_peaks[0][0], wave_peaks[i][1])
		wave_peaks = wave_peaks[i+1:]
		wave_peaks.insert(0, wave)
				
	#去除车牌上的分隔点
	point = wave_peaks[2]
	if point[1] - point[0] < max_wave_dis/3:
		point_img = gray_img[:,point[0]:point[1]]
		if np.mean(point_img) < 255/5:
			wave_peaks.pop(2)
				
	if len(wave_peaks) <= 6:
		print("peak less 2:", len(wave_peaks))

	#调用自定义的函数，分割车牌中的字符
	part_cards = seperate_card(gray_img, wave_peaks)
	########################################组合汉字，去除车牌上的分割点（结束）############################################
	return part_cards




