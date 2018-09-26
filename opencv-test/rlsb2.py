# -*- coding:utf-8 -*-
# OpenCV版本的视频检测
import cv2

def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色

    # OpenCV人脸识别分类器
    classifier = cv2.CascadeClassifier(
        "D:/WorkSpace/python-work/cv/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")

    # 调用识别人脸
    faceRects = classifier.detectMultiScale(gray)

    color = (0, 0, 255)  # 定义绘制颜色
    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            [x, y, w, h] = faceRect
            cv2.rectangle(img, (x, y), (x + w, y + w), color, 1)  # 绘制矩形

    cv2.imshow("Image", img)

# 获取摄像头0表示第一个摄像头
cap = cv2.VideoCapture(0)
while (1):  # 逐帧显示
    ret, img = cap.read()
    #cv2.imshow("Image", img)
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源