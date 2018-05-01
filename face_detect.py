# -*- coding: utf-8 -*-

import sys
import os
from math import pow
from PIL import Image, ImageDraw,ImageFont
#from draw_landmarkfortest import face_Detect
import cv2
import math
import random
import numpy as np
import time

caffe_root = '/home/tas/code/caffe/'
sys.path.insert(0, caffe_root+'python')
# 设置log等级
os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_mode_gpu()


temp_path =  './temp_img/'

#def face_detection(imgFile):
def face_detection(img):
    net_full_conv = caffe.Net('deploy_full_conv.prototxt',
                              'alexnet.caffemodel',
                              caffe.TEST)
    scales = [] # 刻度
    factor = 0.79 # 变换的倍数
    # img = cv2.imread(imgFile)
    # 最大倍数
    largest = min(2, 4000/max(img.shape[0:2]))
    # 最小的边的长度
    minD = largest*min(img.shape[0:2])
    scale = largest
    # 从最大到最小227,获取变换的倍数
    while minD >= 227:
        scales.append(scale)
        scale *= factor
        minD *= factor
    # 存储人脸图
    total_box = []

    # 变换图片
    for scale in scales:
        # fileName = "img_"+str(scale)+'.jpg'
        scale_img = cv2.resize(img, (int((img.shape[0]*scale)), int(img.shape[1]*scale)))
        imTest = scale_img.copy()
        imTest = cv2.cvtColor(imTest, cv2.COLOR_BGR2RGB)
        im = imTest / 255.
        # cv2.imwrite(temp_path+fileName, scale_img)
        # im = caffe.io.load_image(temp_path+fileName)
        # 动态修改数据层的大小?这里为什么时1,0 而不是0,1
        net_full_conv.blobs['data'].reshape(1, 3, scale_img.shape[1], scale_img.shape[0])
        transformer = caffe.io.Transformer({'data':net_full_conv.blobs['data'].data.shape})
        # 减均值，归一化
        transformer.set_mean('data', np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
        # 维度变换 ,cafee默认的时BGR格式，要把RGB(0,1,2)改为BGR(2,0,1)
        transformer.set_transpose('data', (2, 0, 1))
        # 像素
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))

        # 人脸坐标映射
        # 前先传播,映射到原始图像的位置
        out = net_full_conv.forward_all(data=np.asarray(transformer.preprocess('data', im)))
        #out['prob'][0, 1] 0表示类别，1表示概率
        boxes = GenrateBoundingBox(out['prob'][0, 1], scale)
        if(boxes):
            total_box.extend(boxes)
    boxes_nums = np.array(total_box)
    #nms 处理
    true_boxes = nms_average(boxes_nums, 1, 0.2)
    if not true_boxes == []:
        x1,y1,x2,y2 = true_boxes[0][:-1]
        return [x1, x2, y1, y2], img

        cv2.rectangle(img,(int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=5)
        #cv2.imwrite('/home/tas/code/learn/result_img/result.jpg', img)
        cv2.imshow('test', img)
        cv2.waitKey(0)

    return None, img


def GenrateBoundingBox(featureMap, scale):
    boundingBox = []
    stride = 32
    cellSize = 227 #滑动窗口的大小
    for (x, y), prob in np.ndenumerate(featureMap):
        if prob>0.95:
            boundingBox.append([float(stride*y)/scale, float(stride*x)/scale,
                               float(stride * y+ cellSize - 1) / scale, float(stride*x+ cellSize - 1)/scale,
                               prob])

    return boundingBox


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def range_overlap(a_min, a_max, b_min, b_max):

    return (a_min <= b_max) and (b_min <= a_max)

def rect_overlaps(r1,r2):
    return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)

def rect_merge(r1,r2, mergeThresh):

    if rect_overlaps(r1,r2):
        # dist = calculateDistance((r1.left + r1.right)/2, (r1.top + r1.bottom)/2, (r2.left + r2.right)/2, (r2.top + r2.bottom)/2)
        SI= abs(min(r1.right, r2.right) - max(r1.left, r2.left)) * abs(max(r1.bottom, r2.bottom) - min(r1.top, r2.top))
        SA = abs(r1.right - r1.left)*abs(r1.bottom - r1.top)
        SB = abs(r2.right - r2.left)*abs(r2.bottom - r2.top)
        S=SA+SB-SI
        ratio = float(SI) / float(S)
        if ratio > mergeThresh :
            return 1
    return 0
class Rect(object):
    def __init__(self, p1, p2):
        '''Store the top, bottom, left and right values for points
               p1 and p2 are the (corners) in either order
        '''
        self.left   = min(p1.x, p2.x)
        self.right  = max(p1.x, p2.x)
        self.bottom = min(p1.y, p2.y)
        self.top    = max(p1.y, p2.y)

    def __str__(self):
        return "Rect[%d, %d, %d, %d]" % ( self.left, self.top, self.right, self.bottom )
def nms_average(boxes, groupThresh=2, overlapThresh=0.2):
    rects = []
    temp_boxes = []
    weightslist = []
    new_rects = []
    for i in range(len(boxes)):
        if boxes[i][4] > 0.2:
            rects.append([boxes[i,0], boxes[i,1], boxes[i,2]-boxes[i,0], boxes[i,3]-boxes[i,1]])


    rects, weights = cv2.groupRectangles(rects, groupThresh, overlapThresh)

    rectangles = []
    for i in range(len(rects)):

        testRect = Rect( Point(rects[i,0], rects[i,1]), Point(rects[i,0]+rects[i,2], rects[i,1]+rects[i,3]))
        rectangles.append(testRect)
    clusters = []
    for rect in rectangles:
        matched = 0
        for cluster in clusters:
            if (rect_merge( rect, cluster , 0.2) ):
                matched=1
                cluster.left   =  (cluster.left + rect.left   )/2
                cluster.right  = ( cluster.right+  rect.right  )/2
                cluster.top    = ( cluster.top+    rect.top    )/2
                cluster.bottom = ( cluster.bottom+ rect.bottom )/2

        if ( not matched ):
            clusters.append( rect )
    result_boxes = []
    for i in range(len(clusters)):

        result_boxes.append([clusters[i].left, clusters[i].bottom, clusters[i].right, clusters[i].top, 1])

    return result_boxes
if __name__ == '__main__':
    fla = 'img'
    if(fla=='img'):
        faceImg = cv2.imread('11.jpg')
        faceBox, img = face_detection(faceImg)
        cv2.rectangle(img, (faceBox[0], faceBox[2]), (faceBox[1], faceBox[3]), (0, 0, 255), 2)
        cv2.imshow('Contours', img)
        k = cv2.waitKey(0)
      
    else:
        n = 0
        faceBox = []
        cap = cv2.VideoCapture(0)
        while (cap.isOpened()):
            n = n + 1
            # 读取图片 540 960
            ret, img = cap.read()
            if img is None:
                break
            start = time.clock()
            if n%3 == 0:
                faceBox, img = face_detection(img)
                cv2.rectangle(img, (faceBox[0], faceBox[2]), (faceBox[1], faceBox[3]), (0, 0, 255), 2)
                finalImg = img
            if faceBox is None or faceBox==[]:
                finalImg = img
            else:
                elapsed = (time.clock() - start)
                startF = time.clock()
                #finalImg = face_Detect(faceBox, img)
                elapsedF = (time.clock() - startF)
                curStr = "("+str(elapsed)+";"+str(elapsedF)+")"
                cv2.putText(finalImg, curStr, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.imshow('Contours', finalImg)

            k = cv2.waitKey(10)
            if k == 27:
                break
