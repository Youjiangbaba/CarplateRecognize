#coding=utf-8
import numpy as np
import cv2
import os
import time
import numpy as np
import copy
import sys
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

#精确定位，返回颜色（b y g）
def GetCarplate(img_bak,watches):
	color = 0
	for (x, y, w, h) in watches:
		#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
		patch  = img_bak[y:y + h, x:x + w]
		#转为hsv 求h通道，得出蓝色阈值，取该阈值二值化
		patch_hsv = img[y:y + h, x:x + w]
		patch_hsv = cv2.cvtColor(patch_hsv, cv2.COLOR_BGR2HSV)
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
	return img_bak,img_bak,0


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
	plt.figure('竖直方向波峰图')
	plt.plot(x_histogram)
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
	plt.figure('字符所在的区域的车牌图像')
	plt.imshow(gray_img, cmap = 'gray')
								
	#查找水平直方图波峰
	row_num, col_num= gray_img.shape[:2]
	#去掉车牌上下边缘1个像素，避免白边影响阈值判断
	gray_img = gray_img[1:row_num-1]
	#参数axis为0表示压缩列，将每一列的元素相加，将矩阵压缩为一行
	y_histogram = np.sum(gray_img, axis=0)

	plt.figure('水平方向波峰图')
	plt.plot(y_histogram)
	plt.show()	
	
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



#根据颜色信息精确定位车牌位置
def accurate_place(card_img_hsv, color):
	if color == 'b':
		limit1 = blue_min[0]
		limit2 = blue_max[0]
	elif color == 'green':
		limit1 = blue_min[0]
		limit2 = blue_max[0]
	elif color == 'y':
		limit1 = blue_min[0]
		limit2 = blue_max[0]
	#初始车牌区域图像的行数和列数
	row_num, col_num = card_img_hsv.shape[:2]
	xl = col_num
	xr = 0
	yb = 0
	yt = row_num
	row_num_limit = 20
	col_num_limit = col_num * 0.8 if color != "g" else col_num * 0.5#绿色有渐变

	#根据颜色确定车牌图像的上下边界
	for i in range(row_num):
		count = 0
		for j in range(col_num):
			H = card_img_hsv.item(i, j, 0)
			S = card_img_hsv.item(i, j, 1)
			V = card_img_hsv.item(i, j, 2)
			if limit1 < H <= limit2 and 34 < S and 46 < V:
				count += 1

		if count > col_num_limit:
			if yt > i:
				yt = i
			if yb < i:
				yb = i
	#根据颜色确定车牌图像的左右边界
	for j in range(col_num):
		count = 0
		for i in range(row_num):
			H = card_img_hsv.item(i, j, 0)
			S = card_img_hsv.item(i, j, 1)
			V = card_img_hsv.item(i, j, 2)
			if limit1 < H <= limit2 and 34 < S and 46 < V:
				count += 1

		if count > row_num - row_num_limit:
			if xl > j:
				xl = j
			if xr < j:
				xr = j
				
	return xl, xr, yb, yt, limit1, limit2


def find_rightcounter(car_pic,color):
	if type(car_pic) == type(""):
		#如果输入是一个字符串，则从它指向的路径读取图片
		img = imreadex(car_pic)
	else:
		#否则，认为输入就是一张图片
		img = car_pic
	oldimg = img
	original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	pic_hight, pic_width = img.shape[:2]

	#如果输入图像过大，则调整图像大小
	#if pic_width > 500:
	#	resize_rate = 500 / pic_width
	#	img = cv2.resize(img, (500, int(pic_hight*resize_rate)), interpolation=cv2.INTER_AREA)
	#高斯模糊去噪
	blur = 3
	if blur > 0:
		img = cv2.GaussianBlur(img, (blur, blur), 0)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	#找到图像边缘
	#ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#大津阈值
	#img_edge = cv2.Canny(img_thresh, 50, 200)#canny算子提取边缘
	
	ret, img_edge = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#大津阈值
	#img_edge = cv2.bitwise_not(img_edge)
	cv2.imshow("1",img_edge)	

	#边缘整体化，使用开运算和闭运算让图像边缘成为一个整体
	kernel = np.ones((20, 20), np.uint8)
	img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
	img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

	

	#查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
	#coutours里面保存的是轮廓里面的点，cv2.CHAIN_APPROX_SIMPLE表示：只保存角点
	image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#检测物体轮廓
	tmp1 = image
	tmp1 = cv2.cvtColor(tmp1, cv2.COLOR_GRAY2BGR)#灰度图转化为彩色图

	#参数-1表示画出所有轮廓
	#参数(0, 255, 0)表示为轮廓上色的画笔颜色为绿色
	#参数2表示画笔的粗细为2
	cv2.drawContours(tmp1, contours, -1, (0, 255, 0), 2)	# 画出分割出来的轮廓

	#挑选出面积大于Min_Area的轮廓
	contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
	#print('len(contours)', len(contours))
	tmp2 = image
	tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_GRAY2BGR)#灰度图转化为彩色图
	cv2.drawContours(tmp2, contours, -1, (0, 255, 0), 2)	

	cv2.imshow("contours",tmp2)
	##########################################################计算车牌可能出现矩形区域（开始）###############################################
	car_contours = []
	tmp3 = image
	tmp3 = cv2.cvtColor(tmp3, cv2.COLOR_GRAY2RGB)#灰度图转化为彩色图
	tmp4 = copy.deepcopy(oldimg)				#import copy
	tmp4 = cv2.cvtColor(tmp4, cv2.COLOR_BGR2RGB)
	for cnt in contours:
		#使用cv2.minAreaRect函数生成每个轮廓的最小外界矩形
		#输入为表示轮廓的点集
		#返回值rect中包含最小外接矩形的中心坐标，宽度高度和旋转角度（但是这里的宽度和高度不是按照其长度来定义的）
		rect = cv2.minAreaRect(cnt)
		area_width, area_height = rect[1]
		#做一定调整，保证宽度大于高度
		if area_width < area_height:
			area_width, area_height = area_height, area_width
		wh_ratio = area_width / area_height
		#print(wh_ratio)

		#要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
		if wh_ratio > 2 and wh_ratio < 5.5:
			car_contours.append(rect)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(tmp3, [box], -1, (0, 255, 0), 2)
			cv2.drawContours(tmp4, [box], -1, (0, 255, 0), 2)

	######################################################将倾斜的矩形调整为不倾斜（开始）###################################################
		# print("精确定位")
	print(rect)
	card_imgs = []
	#矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
	for rect in car_contours:
		#调整角度，使得矩形框左高右低
		#0度和1度之间的所有角度当作1度处理,-1度和0度之间的所有角度当作-1度处理
		#这个处理是必要的，如果不做这个处理的话，后面仿射变换可能会得到一张全灰的图片
		#因为如果角度接近于0,那么矩形四个角点中任意两个角点，其某一个坐标非常接近，这种
		#情况下，哪个角点在最上边，哪个角点在最下边，哪个角点在最左边，哪个角点在最右边，就
		#没有了很强的区分度，所以仿射变换控制点的对应关系，很可能出现错配，造成仿射变换失败
		if rect[2] > -1:
			angle = -1
		else:
			angle = rect[2]
		
		#扩大范围，避免车牌边缘被排除
		rect = (rect[0], (rect[1][0]+5, rect[1][1]+5), angle)
		box = cv2.boxPoints(rect)

		#bottom_point:矩形框4个角中最下面的点
		#right_point:矩形框4个角中最右边的点
		#left_point：矩形框4个角中最左边的点
		#top_point:矩形框4个角中最上面的点
		bottom_point = right_point = [0, 0]
		left_point = top_point = [pic_width, pic_hight]
		for point in box:
			if left_point[0] > point[0]:
				left_point = point
			if top_point[1] > point[1]:
				top_point = point 
			if bottom_point[1] < point[1]:
				bottom_point = point
			if right_point[0] < point[0]:
				right_point = point

		#这里需要注意的是：cv2.boxPoints检测矩形，返回值中角度的范围是[-90, 0]，所以该函数中并不是长度大的作为底，长度
		#小的作为高，而是以从x轴逆时针旋转，最先到达的边为底，另一条边为高
		#这里为了矫正图像所作的仿射变换只能处理小角度，若角度过大，畸变很严重
		#在该程序里，没有对矩形做旋转，然后再仿射变换，而是直接做仿射变换，所以只有当带识别图片中车牌位于水平位置附近时，
		#才能正确识别，而当车牌绕着垂直于车牌的轴有较大转动时，识别就会失败
		if left_point[1] <= right_point[1]:#正角度
			new_right_point = [right_point[0], bottom_point[1]]
			pts2 = np.float32([left_point, bottom_point, new_right_point])
			pts1 = np.float32([left_point, bottom_point, right_point])
			#用3个控制点进行仿射变换
			M = cv2.getAffineTransform(pts1, pts2)
			dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))

			point_limit(new_right_point)
			point_limit(bottom_point)
			point_limit(left_point)
			card_img = dst[int(left_point[1]):int(bottom_point[1]), int(left_point[0]):int(new_right_point[0])]
			card_imgs.append(card_img)
		elif left_point[1] > right_point[1]:#负角度
			new_left_point = [left_point[0], bottom_point[1]]
			pts2 = np.float32([new_left_point, bottom_point, right_point])
			pts1 = np.float32([left_point, bottom_point, right_point])
			#仿射变换
			M = cv2.getAffineTransform(pts1, pts2)
			dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))

			point_limit(right_point)
			point_limit(bottom_point)
			point_limit(new_left_point)
			card_img = dst[int(right_point[1]):int(bottom_point[1]), int(new_left_point[0]):int(right_point[0])]
			card_imgs.append(card_img)
		######################################################将倾斜的矩形调整为不倾斜（结束）########################################

		########################################################根据车牌颜色再定位，缩小边缘非车牌边界（开始）#####################################
		colors = []
		#enumerate是python的内置函数，对于一个可遍历的对象，enumerate将其组成一个索引序列，利用它可以同时获得索引和值
		for card_index,card_img in enumerate(card_imgs):
			if card_img is None:
				continue
			card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
			#有转换失败的可能，原因来自于上面矫正矩形出错

			row_num, col_num= card_img_hsv.shape[:2]
			xl, xr, yb, yt, limit1, limit2 = accurate_place(card_img_hsv, color)
			if yt == yb and xl == xr:
				continue
			need_accurate = False
			if yt >= yb:
				yt = 0
				yb = row_num
				need_accurate = True
			if xl >= xr:
				xl = 0
				xr = col_num
				need_accurate = True
			card_imgs[card_index]  = card_img[yt:yb, xl:xr] if color != "g" or yt < (yb-yt)//4 else card_img[yt-(yb-yt)//4:yb, xl:xr]
			if need_accurate:#可能x或y方向未缩小，需要再试一次
				card_img = card_imgs[card_index]
				card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
				xl, xr, yb, yt, limit1, limit2 = accurate_place(card_img_hsv, color)
				if yt == yb and xl == xr:
					continue
				if yt >= yb:
					yt = 0
					yb = row_num
				if xl >= xr:
					xl = 0
					xr = col_num
			card_imgs[card_index]  = card_img[yt:yb, xl:xr] if color != "g" or yt < (yb-yt)//4 else card_img[yt-(yb-yt)//4:yb, xl:xr]
			cv2.imshow("ok"+str(card_index),card_imgs[card_index])
		################################################根据车牌颜色再定位，缩小边缘非车牌边界（结束）#####################################

if __name__ == '__main__':
	watch_cascade = cv2.CascadeClassifier('./out-long/cascade.xml')
	count = 0;
	extend_scale = 1;
	shape = [140,180]
	path = "./general_test"
	char_imgs = []
	for parent,dirnames,filenames in os.walk("test_image/"):                     # /home/jiang/Repositories/2.15车牌识别/licence-plate-recognition-master/src/test/


		for filename in filenames:
		# while(1):
		#     filename = filenames[np.random.randint(0,len(filenames))]
			path = os.path.join(parent,filename)
			if path.endswith(".jpg"):
				img = cv2.imread(path)
				img_bak  = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
				img_bak = cv2.GaussianBlur(img_bak,(3,3),2)
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				t0 = time.time()
				watches = watch_cascade.detectMultiScale(gray, 1.1, 5,minSize=(114 , 27))            #opencv LBP特征分类器,识别带车牌区域
				print time.time() - t0
				mask0,extend_patch,color = GetCarplate(img_bak,watches)
				if color>0:
					print color
					cv2.imshow("extend_patch",extend_patch)
					#cv2.imshow("mask0",mask0)
					#extend_patch_resize = cv2.resize(extend_patch, (2*extend_patch.shape[1], 2*extend_patch.shape[0]))
					#find_rightcounter(extend_patch,color)
					carplate_img = morphology_deal(extend_patch)
					char_imgs = Character_separation(carplate_img)
					cv2.imshow("result",carplate_img)
					for char_i,result_img in enumerate(char_imgs):
						cv2.imshow("char"+str(char_i),result_img)
	###############################################################################################################################3
				count += 1
				#cv2.imshow("img"+str(count),img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
