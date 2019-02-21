#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import os
import json
import time
from numpy.linalg import norm
from matplotlib import pyplot as plt
import copy
import Get_CarPlate as cp


SZ = 20          #训练图片长宽
PROVINCE_START = 1000
#不能保证包括所有省份
provinces = [
"zh_cuan", "川",
"zh_e", "鄂",
"zh_gan", "赣",
"zh_gan1", "甘",
"zh_gui", "贵",
"zh_gui1", "桂",
"zh_hei", "黑",
"zh_hu", "沪",
"zh_ji", "冀",
"zh_jin", "津",
"zh_jing", "京",
"zh_jl", "吉",
"zh_liao", "辽",
"zh_lu", "鲁",
"zh_meng", "蒙",
"zh_min", "闽",
"zh_ning", "宁",
"zh_qing", "靑",
"zh_qiong", "琼",
"zh_shan", "陕",
"zh_su", "苏",
"zh_sx", "晋",
"zh_wan", "皖",
"zh_xiang", "湘",
"zh_xin", "新",
"zh_yu", "豫",
"zh_yu1", "渝",
"zh_yue", "粤",
"zh_yun", "云",
"zh_zang", "藏",
"zh_zhe", "浙"
]

#来自opencv的sample，用于svm训练
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

#求HOG特征
def preprocess_hog(digits):
	samples = []
	for img in digits:
		#对输入中的每一幅图像求梯度直方图
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bin = np.int32(bin_n*ang/(2*np.pi))
		bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)
		
		# transform to Hellinger kernel
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps
		
		samples.append(hist)
	return np.float32(samples)


class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)  
	def save(self, fn):
		self.model.save(fn)

class SVM(StatModel):
	def __init__(self, C = 1, gamma = 0.5):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
	#训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
	#字符识别
	def predict(self, samples):
		r = self.model.predict(samples)
		return r[1].ravel()

class CardPredictor:
	def __init__(self):
		#车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
		f = open('config.js')
		j = json.load(f)
		for c in j["config"]:
			if c["open"]:
				self.cfg = c.copy()
				break
		else:
			raise RuntimeError('没有设置有效配置参数')

	def __del__(self):
		self.save_traindata()
	def train_svm(self):
		#识别英文字母和数字
		self.model = SVM(C=1, gamma=0.5)
		#识别中文
		self.modelchinese = SVM(C=1, gamma=0.5)
		if os.path.exists("svm.dat"):
			self.model.load("svm.dat")
		else:
			chars_train = []
			chars_label = []
			
			for root, dirs, files in os.walk("train\\chars2"):
				if len(os.path.basename(root)) > 1:
					continue
				root_int = ord(os.path.basename(root))
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					#chars_label.append(1)
					chars_label.append(root_int)
			
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)
			#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
			chars_label = np.array(chars_label)
			print(chars_train.shape)
			self.model.train(chars_train, chars_label)
		if os.path.exists("svmchinese.dat"):
			self.modelchinese.load("svmchinese.dat")
		else:
			chars_train = []
			chars_label = []
			for root, dirs, files in os.walk("train\\charsChinese"):
				if not os.path.basename(root).startswith("zh_"):
					continue
				pinyin = os.path.basename(root)
				index = provinces.index(pinyin) + PROVINCE_START + 1 #1是拼音对应的汉字
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					#chars_label.append(1)
					chars_label.append(index)
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)
			#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
			chars_label = np.array(chars_label)
			print(chars_train.shape)
			self.modelchinese.train(chars_train, chars_label)

	def save_traindata(self):
		if not os.path.exists("svm.dat"):
			self.model.save("svm.dat")
		if not os.path.exists("svmchinese.dat"):
			self.modelchinese.save("svmchinese.dat")

	def predict(self,part_cards):
		predict_result = []
		for i, part_card in enumerate(part_cards):
		# print(part_card)
		#可能是固定车牌的铆钉
			if np.mean(part_card) < 255/5:
				print("a point")
				continue
			part_card_old = part_card
			w = abs(part_card.shape[1] - SZ)//2#//运算符取除法结果的最小整数
							
			#给图像扩充边界cv2.copyMakeBorder(src,top, bottom, left, right ,borderType,value)
			part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])#左右边界用黑色背景扩充
			part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)#将分割出来的图像调整为设定的训练图像的尺寸			
			ret, part_card = cv2.threshold(part_card, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

			#title = "分割出来的字符" + str(i + 1)
			#plt.figure(title)
			#plt.imshow(part_card, cmap = 'gray')
			#plt.axis('off')
			# plt.show()

			#part_card = deskew(part_card)
			#根据分割出的字符图像，计算HOG特征
			part_card = preprocess_hog([part_card])

			if i == 0:
				#如果是第一个字符，则识别汉字
				resp = self.modelchinese.predict(part_card)
				charactor = provinces[int(resp[0]) - PROVINCE_START]
			else:
				#如果不是第一个字符，则识别字母和数字
				resp = self.model.predict(part_card)
				charactor = chr(resp[0])

			#判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
			if charactor == "1" and i == len(part_cards)-1:
				if part_card_old.shape[0]/part_card_old.shape[1] >= 7:#1太细，认为是边缘
					continue

			predict_result.append(charactor)
		return predict_result	#识别到的字符



if __name__ == '__main__':
	c = CardPredictor()
	c.train_svm()

	watch_cascade = cv2.CascadeClassifier('./out-long/cascade.xml')
	count = 0;
	path = "./general_test"
	char_imgs = []
	for parent,dirnames,filenames in os.walk("test_image/"):                     # /home/jiang/Repositories/2.15车牌识别/licence-plate-recognition-master/src/test/
		for filename in filenames:
			path = os.path.join(parent,filename)
			if path.endswith(".jpg"):
				if count > 10:
					pass
				img = cv2.imread(path)
				#img_bak  = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
				#img_bak = cv2.GaussianBlur(img_bak,(3,3),2)
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				t0 = time.time()
				watches = watch_cascade.detectMultiScale(gray, 1.1, 5,minSize=(114 , 27))            #opencv LBP特征分类器,识别带车牌区域
				print time.time() - t0
				#车牌定位 
				mask0,extend_patch,color = cp.GetCarplate(img,watches)                   
				if color>0:
					print color
					cv2.imshow("extend_patch",extend_patch)
					#二值化 字符显著   （如果增加仿射变换，矫正图片）
					carplate_img = cp.morphology_deal(extend_patch)
					#字符分割
					char_imgs = cp.Character_separation(carplate_img)
					print (c.predict(char_imgs))
					cv2.imshow("result",carplate_img)
					#for char_i,result_img in enumerate(char_imgs):
						#cv2.imshow("char"+str(char_i),result_img)
						
				count += 1
				cv2.waitKey(0)
				cv2.destroyAllWindows()

