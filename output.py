#encoding：utf-8
import numpy as np
import cv2
import os
# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    child_file_name = []
    full_child_file_list = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        #child = child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        full_child_file_list.append(child)
        child_file_name.append(allDir)
    return full_child_file_list, child_file_name


def mkdir(path):  # 判断是否存在指定文件夹，不存在则创建
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print('mkdir' + path)
        return True
    else:
        return False

img_dir = './ge_0_5/'
new_img_dir = './ge_0_5/processlv/'
mkdir(new_img_dir)
full_child_file_list, child_file_name = eachFile(img_dir)
for num,imgName in enumerate(full_child_file_list):

    newImgName = new_img_dir + child_file_name[num]
    img = cv2.imread(imgName )  # 直接读为灰度图像
    #BGR转化为HSV
    HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    color = [
    ([0,0,0], [120, 120, 120])  # 蓝色范围~这个是我自己试验的范围，可根据实际情况自行调整~注意：数值按[b,g,r]排布
]
# 如果color中定义了几种颜色区间，都可以分割出来
    for (lower, upper) in color:
        # 创建NumPy数组
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限
        # 根据阈值找到对应颜色
        mask = cv2.inRange(HSV, lower, upper)    #查找处于范围区间的
        mask = 255-mask                          #留下铝材区域
        output = cv2.bitwise_and(img, img, mask=mask)    #获取铝材区域
        #bgroutput = cv2.cvtColor(output,cv2.COLOR_HSV2BGR)
        # 展示图片
        #cv2.imshow("images", np.hstack([img, output]))
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(mask.shape)
        # print(mask[0])
        # print(len(contours))
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    for i in contours:
        print(cv2.contourArea(i))  # 计算缺陷区域面积
    x, y, w, h = cv2.boundingRect(contours[0])  # 画矩形框
    #print((contours[0]))
    maxArea = 0
    for numcontours, contour in enumerate(contours):
        if (cv2.contourArea(contour )>maxArea):
            maxArea = cv2.contourArea( contour )
            x, y, w, h = cv2.boundingRect(contour)

    # img1 = img[y:y+h,x: x+ w]
    # print(x)
    #cv2.drawContours(img, contours, -1, (0, 0, 255), 10)
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
    #cv.imwrite(show_result_path, match_img_color)
    #cv2.imshow("detect",img)
    #cv2.waitKey(0)
    cv2.imwrite(newImgName,output)


