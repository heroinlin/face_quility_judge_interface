import math

import cv2
import numpy as np
from skimage.measure import compare_ssim


# brenner梯度函数计算
def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 2):
        for y in range(0, shape[1]):
            out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
    return out


# Laplacian梯度函数计算
def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    return cv2.Laplacian(img, cv2.CV_64F).var()


# SMD梯度函数计算
def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0] - 1):
        for y in range(0, shape[1]):
            out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
            out += math.fabs(int(img[x, y] - int(img[x + 1, y])))
    return out


# SMD2梯度函数计算
def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            out += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * \
                   math.fabs(int(img[x, y] - int(img[x, y + 1])))
    return out


# 方差函数计算
def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            out += (img[x, y] - u) ** 2
    return out


# energy函数计算
def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            out += ((int(img[x + 1, y]) - int(img[x, y])) ** 2) * \
                   ((int(img[x, y + 1] - int(img[x, y]))) ** 2)
    return out


# Vollath函数计算
def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0] * shape[1] * (u ** 2)
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1]):
            out += int(img[x, y]) * int(img[x + 1, y])
    return out


# entropy函数计算
def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    count = np.shape(img)[0] * np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i] != 0:
            out -= p[i] * math.log(p[i] / count) / count
    return out


# NRS 梯度结构相似度
class NRSS():
    def __init__(self):
        pass

    def gauseBlur(self, img):
        img_Guassian = cv2.GaussianBlur(img, (7, 7), 0)
        return img_Guassian

    def sobel(self, img):
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return dst

    def getBlock(self, image):
        Ir = self.gauseBlur(image)
        G = self.sobel(image)
        Gr = self.sobel(Ir)
        blocksize = 8
        (h, w) = G.shape
        G_blk_list = []
        Gr_blk_list = []
        sp = 6
        for i in range(sp):
            for j in range(sp):
                G_blk = G[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
                Gr_blk = Gr[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
                G_blk_list.append(G_blk)
                Gr_blk_list.append(Gr_blk)
        sum = 0
        for i in range(sp * sp):
            mssim = compare_ssim(G_blk_list[i], Gr_blk_list[i])
            sum = mssim + sum
        nrss = 1 - sum / (sp * sp * 1.0)
        return nrss

    def __call__(self, image):
        return self.getBlock(image)


def main(img1, img2):
    print('Brenner', brenner(img1), brenner(img2))
    print('Laplacian', Laplacian(img1), Laplacian(img2))
    print('SMD', SMD(img1), SMD(img2))
    print('SMD2', SMD2(img1), SMD2(img2))
    print('Variance', variance(img1), variance(img2))
    print('Energy', energy(img1), energy(img2))
    print('Vollath', Vollath(img1), Vollath(img2))
    print('Entropy', entropy(img1), entropy(img2))
    nrss = NRSS()
    print('NRSS', nrss(img1), nrss(img2))


if __name__ == '__main__':
    # 读入原始图像
    import os

    root_dir = os.path.split(os.path.realpath(__file__))[0]
    image_path = r"F:\tmp\person_search\face\88_1.jpg"
    print(image_path)
    image = cv2.imread(image_path)
    # # left, top, width, height
    # boxes = [146, 193, 371, 371]
    # # 灰度化处理
    # image = image[boxes[1]:boxes[1] + boxes[3], boxes[0]:boxes[0] + boxes[2], :]
    image = cv2.resize(image, (112, 112))
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    img2 = cv2.resize(img2, (image.shape[1], image.shape[0]))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    main(img1, img2)
