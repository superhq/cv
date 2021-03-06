import cv2
from keras.models import load_model
import numpy as np

filepath = "../photos/couple.jpg"
img = cv2.imread(filepath)  # 读取图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色

# OpenCV人脸识别分类器
classifier = cv2.CascadeClassifier("D:/WorkSpace/python-work/cv/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")

# 调用识别人脸
faceRects = classifier.detectMultiScale(gray)

#性别分类器
gender_classifier = load_model(
    "classifier/gender_models/simple_CNN.81-0.96.hdf5")
gender_labels = {0: 'women', 1: 'man'}


font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 0, 255)  # 定义绘制颜色
if len(faceRects):  # 大于0则检测到人脸
    for faceRect in faceRects:  # 单独框出每一张人脸
        [x, y, w, h] = faceRect

        face = img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, 0)
        face = face / 255.0
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]
        print(gender)
        cv2.rectangle(img, (x, y), (x + w, y + w), color, 1)  # 绘制矩形
        img = cv2.putText(img, gender, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


#cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow("Image", img)  # 显示图像
cv2.waitKey(0)
cv2.destroyAllWindows()  # 释放所有的窗体资源