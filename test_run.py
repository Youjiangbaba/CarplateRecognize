#coding=utf-8
import numpy as np
import cv2
import os
import time
import numpy as np

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
		mask_b=cv2.bitwise_and(mask_b, mask_b)
		mask_g=cv2.inRange(patch_hsv,  green_min,  green_max)
		mask_g=cv2.bitwise_and(mask_g, mask_g)
		mask_y=cv2.inRange(patch_hsv,  yello_min,  yello_max)
		mask_y=cv2.bitwise_and(mask_y, mask_y)
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
	

def morphology_deal(img):

	# 再次膨胀，让轮廓明显一些
	#dilation2 = cv2.dilate(dilation, element2,iterations = 1)
	if 1:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)       #大津阈值法
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



watch_cascade = cv2.CascadeClassifier('./out-long/cascade.xml')
count = 0;
extend_scale = 1;

shape = [140,180]
green_min = np.array([35, 100, 46]) 
green_max = np.array([70, 255, 255])

yello_min = np.array([20, 100, 46]) 
yello_max = np.array([30, 255, 255])

blue_min = np.array([100, 100, 46]) 
blue_max = np.array([124, 255, 255]) 

white_min = np.array([40, 40, 100]) 
white_max = np.array([150, 200, 255]) 
path = "./general_test"
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
				cv2.imshow("mask0",mask0)
				#extend_patch_resize = cv2.resize(extend_patch, (2*extend_patch.shape[1], 2*extend_patch.shape[0]))

				carplate_img = morphology_deal(extend_patch)
				cv2.imshow("result",carplate_img)
###############################################################################################################################3

			count += 1
			#cv2.imshow("img"+str(count),img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
