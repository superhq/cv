import cv2
import numpy as np
from matplotlib import pyplot as plt


def showimg():
    img = cv2.imread('../photos/people.jpg', cv2.IMREAD_UNCHANGED)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showimg2():
    img = cv2.imread('../photos/people.jpg', cv2.IMREAD_UNCHANGED)  # BGR mode
    # 手工转换
    # b,g,r = cv2.split(img)
    # img = cv2.merge([r, g, b])

    # 用函数进行转换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # matplotlib的图像采用RGB模式
    # img = plt.imread('../photos/people.jpg') #RGB mode
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.show()


def capture_video():
    cap = cv2.VideoCapture(0)

    # 获取和设置帧的大小
    print('{0} x {1}'.format(cap.get(3), cap.get(4)))
    cap.set(3, 320)
    cap.set(4, 240)

    while True:
        ret, frame = cap.read()
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def show_video():
    cap = cv2.VideoCapture('output.avi')

    # 获取和设置帧的大小
    print('{0} x {1}'.format(cap.get(3), cap.get(4)))
    cap.set(3, 320)
    cap.set(4, 240)

    while True:
        ret, frame = cap.read()
        cv2.imshow('video', frame)
        if cv2.waitKey(25) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def capture_and_write_video():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 翻转
            frame = cv2.flip(frame, 0)
            out.write(frame)
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


show_video()
