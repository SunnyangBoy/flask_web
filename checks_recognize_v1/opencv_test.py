# （基于透视的图像矫正）
import cv2
import math
import numpy as np


def Img_Outline(input_dir):
    original_img = cv2.imread(input_dir)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)  # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY_INV)  # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 定义矩形结构元素
    # closed = cv2.dilate(RedThresh, kernel)
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)  # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 开运算（去噪点）
    #opened = cv2.Canny(opened, 0, 255)  # 50是最小阈值,80是最大阈值
    return original_img, gray_img, RedThresh, closed, opened

'''
def findContours_img(original_img, opened):
    image, contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[1]  # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)  # 获取包围盒（中心点，宽高，旋转角度）
    box = np.int0(cv2.boxPoints(rect))  # box
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    print("box[0]:", box[0])
    print("box[1]:", box[1])
    print("box[2]:", box[2])
    print("box[3]:", box[3])
    return box, draw_img

def findContours_img(original_img, opened):
    initHei, initWid = opened.shape[:2]

    binary, contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area = -1
    for contour in contours:
        # cv2.drawContours(img, contour, -1, (255, 255, 0), 3)

        # 获取最小包围矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(img, [ansBox], 0, (255, 0, 0), 3)

        # 中心坐标
        x, y = rect[0]
        # cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), 8)

        # 角度:[-90,0)
        angle = rect[2]

        # 长和宽
        a, b = rect[1]  # a是水平轴逆时针选择遇到的第一条边
        if a >= b:  # 竖直放置，应当调整到0度
            wd = a
            ht = b
        else:  # 正常放置，调整到-90度
            wd = b
            ht = a

        if x > int(initWid * 0.25) and x < int(initWid * 0.75) and y > int(initHei * 0.25) and y < int(initHei * 0.75):
            if ht and wd / ht < 3 and wd / ht > 1:
                if wd * ht > area:
                    ansBox = box
                    area = wd * ht
                    print('width=', wd, 'height=', ht, 'x=', x, 'y=', y, 'angle=', angle)

    print(ansBox)
    draw_img = cv2.drawContours(original_img, [ansBox], 0, (0, 0, 255), 3)
    return ansBox, draw_img
'''

def findContours_img(original_img, opened):
    original_img = original_img.copy()
    vertices = []

    height, width = original_img.shape[:2]
    upleft_lr = int(-1/(height/width))
    upright_lr = int(-1/(-height/width))
    dowleft_lr = upright_lr
    dowright_lr = upleft_lr

    vertice1 = (0, 0)
    for h in range(height):
        flag = 1
        for x in range(width):
            y = (upleft_lr * x) + h
            if y < height and y > 0:
                if opened[y][x] > 0:
                    vertice1 = (x, y)
                    flag = 0
                    break
        if flag == 0:
            break
    print('vertice1: ', vertice1)
    vertices.append(vertice1)
    draw_img = cv2.circle(original_img, vertice1, 5, [0, 0, 255])


    vertice2 = (width-1, 0)
    for h in range(height):
        flag = 1
        for x in range(width):
            y = h - (upright_lr * x)
            if y < height and y > 0:
                if opened[y][width-1-x] > 0:
                    vertice2 = (width-1-x, y)
                    flag = 0
                    break
        if flag == 0:
            break
    print('vertice2: ', vertice2)
    vertices.append(vertice2)
    draw_img = cv2.circle(draw_img, vertice2, 5, [0, 0, 255])


    vertice4 = (width-1, height-1)
    for h in range(height)[::-1]:
        flag = 1
        for x in range(width):
            y = h - (dowright_lr * x)
            if y < height and y > 0:
                if opened[y][width - 1 - x] > 0:
                    vertice4 = (width - 1 - x, y)
                    flag = 0
                    break
        if flag == 0:
            break
    print('vertice4: ', vertice4)
    vertices.append(vertice4)
    draw_img = cv2.circle(draw_img, vertice4, 5, [0, 0, 255])


    vertice3 = (0, height-1)
    for h in range(height)[::-1]:
        flag = 1
        for x in range(width):
            y = (dowleft_lr * x) + h
            if y < height and y > 0:
                if opened[y][x] > 0:
                    vertice3 = (x, y)
                    flag = 0
                    break
        if flag == 0:
            break
    print('vertice3: ', vertice3)
    vertices.append(vertice3)
    draw_img = cv2.circle(draw_img, vertice3, 5, [0, 0, 255])

    vertices.append(vertice1)
    return np.array(vertices), draw_img


def Perspective_transform(box, original_img):
    # 获取画框宽高(x=orignal_W, y=orignal_H)
    orignal_W = math.ceil(np.sqrt((box[3][1] - box[2][1]) ** 2 + (box[3][0] - box[2][0]) ** 2))
    orignal_H = math.ceil(np.sqrt((box[3][1] - box[0][1]) ** 2 + (box[3][0] - box[0][0]) ** 2))

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[2], box[3], box[0], box[1]])
    pts2 = np.float32(
        [[int(orignal_W + 1), int(orignal_H + 1)], [0, int(orignal_H + 1)], [0, 0], [int(orignal_W + 1), 0]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_img = cv2.warpPerspective(original_img, M, (int(orignal_W + 3), int(orignal_H + 1)))

    return result_img


if __name__ == "__main__":
    input_dir = "/Users/zhuzhenyang/Desktop/receipt_img_tm1_02868.jpg"
    original_img, gray_img, RedThresh, closed, opened = Img_Outline(input_dir)
    box, draw_img = findContours_img(original_img, opened)
    draw_img = cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 3)

    '''
    minLineLength = 50
    maxLineGap = 100
    #lines = cv2.HoughLinesP(opened, 0.8, np.pi / 180, 10, minLineLength, maxLineGap)

    lines = cv2.HoughLinesP(opened, 1, np.pi / 180, 80, minLineLength, maxLineGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(opened, (x1, y1), (x2, y2), (0, 255, 0), 2)


    lines = cv2.HoughLines(opened, 1, np.pi / 180, 100)  # 这里对最后一个参数使用了经验型的值
    for line in lines:
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        print(rho)
        print(theta)
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
            # 该直线与最后一行的焦点
            pt2 = (int((rho - opened.shape[0] * np.sin(theta)) / np.cos(theta)), opened.shape[0])
            cv2.line(opened, pt1, pt2, (255),2)  # 绘制一条白线
        else:  # 水平直线
            pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
            # 该直线与最后一列的交点
            pt2 = (opened.shape[1], int((rho - opened.shape[1] * np.cos(theta)) / np.sin(theta)))
            cv2.line(opened, pt1, pt2, (255), 2)  # 绘制一条直线
    '''

    result_img = Perspective_transform(box, original_img)
    cv2.imshow("original", original_img)
    cv2.imshow("RedThresh", RedThresh)
    cv2.imshow("gray", gray_img)
    cv2.imshow("closed", closed)
    cv2.imshow("opened", opened)
    cv2.imshow("draw_img", draw_img)
    cv2.imshow("result_img", result_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

