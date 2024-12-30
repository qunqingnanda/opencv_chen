import cv2
import numpy as np
import pytesseract
from tkinter import Tk, Button, Frame, Label
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageDraw, ImageFont
import threading
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 预处理
def imgProcess(path):
    img = cv2.imread(path)
    # 统一规定大小
    img = cv2.resize(img, (640, 480))
    # 高斯模糊
    img_Gas = cv2.GaussianBlur(img, (5, 5), 0)
    # RGB通道分离
    img_B = cv2.split(img_Gas)[0]
    img_G = cv2.split(img_Gas)[1]
    img_R = cv2.split(img_Gas)[2]
    # 读取灰度图和HSV空间图
    img_gray = cv2.cvtColor(img_Gas, cv2.COLOR_BGR2GRAY)
    img_HSV = cv2.cvtColor(img_Gas, cv2.COLOR_BGR2HSV)
    return img, img_Gas, img_B, img_G, img_R, img_gray, img_HSV


# 初步识别
def preIdentification(img_gray, img_HSV, img_B, img_R):
    # 将 HSV 通道转换为 int16，避免溢出
    h_channel = img_HSV[:, :, 0].astype(np.int16)

    # 条件过滤
    mask = (np.abs(h_channel - 115) < 15) & (img_B > 70) & (img_R < 40)

    # 应用条件到灰度图
    img_gray[mask] = 255
    img_gray[~mask] = 0

    # 定义核
    kernel_small = np.ones((3, 3))
    kernel_big = np.ones((7, 7))

    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)  # 高斯平滑
    img_di = cv2.dilate(img_gray, kernel_small, iterations=5)  # 腐蚀5次
    img_close = cv2.morphologyEx(img_di, cv2.MORPH_CLOSE, kernel_big)  # 闭操作
    img_close = cv2.GaussianBlur(img_close, (5, 5), 0)  # 高斯平滑
    _, img_bin = cv2.threshold(img_close, 100, 255, cv2.THRESH_BINARY)  # 二值化
    return img_bin



# 定位
def fixPosition(img, img_bin):
    # 检测所有外轮廓，只留矩形的四个顶点
    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 形状及大小筛选校验
    det_x_max = 0
    det_y_max = 0
    num = 0
    for i in range(len(contours)):
        x_min = np.min(contours[i][:, :, 0])
        x_max = np.max(contours[i][:, :, 0])
        y_min = np.min(contours[i][:, :, 1])
        y_max = np.max(contours[i][:, :, 1])
        det_x = x_max - x_min
        det_y = y_max - y_min
        if (det_x / det_y > 1.8) and (det_x > det_x_max) and (det_y > det_y_max):
            det_y_max = det_y
            det_x_max = det_x
            num = i
    # 获取最可疑区域轮廓点集
    points = np.array(contours[num][:, 0])
    return points


# img_lic_canny = cv2.Canny(img_lic_bin, 100, 200)
# 获取四个顶点坐标
def findVertices(points):
    # 获取最小外接矩阵，中心点坐标，宽高，旋转角度
    rect = cv2.minAreaRect(points)
    # 获取矩形四个顶点，浮点型
    box = cv2.boxPoints(rect)
    # 取整
    box = np.int64(box)
    # 获取四个顶点坐标
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    # 上下左右四个点坐标
    vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y],
                         [right_point_x, right_point_y]])
    return vertices, rect


# 图像倾斜校正
def tiltCorrection(vertices, rect):
    # 畸变情况1
    if rect[2] > -45:
        new_right_point_x = vertices[0, 0]
        new_right_point_y = int(
            vertices[1, 1] - (vertices[0, 0] - vertices[1, 0]) / (vertices[3, 0] - vertices[1, 0]) * (
                        vertices[1, 1] - vertices[3, 1]))
        new_left_point_x = vertices[1, 0]
        new_left_point_y = int(
            vertices[0, 1] + (vertices[0, 0] - vertices[1, 0]) / (vertices[0, 0] - vertices[2, 0]) * (
                        vertices[2, 1] - vertices[0, 1]))
        # 校正后的四个顶点坐标
        point_set_1 = np.float32([[440, 0], [0, 0], [0, 140], [440, 140]])
    # 畸变情况2
    elif rect[2] < -45:
        new_right_point_x = vertices[1, 0]
        new_right_point_y = int(
            vertices[0, 1] + (vertices[1, 0] - vertices[0, 0]) / (vertices[3, 0] - vertices[0, 0]) * (
                        vertices[3, 1] - vertices[0, 1]))
        new_left_point_x = vertices[0, 0]
        new_left_point_y = int(
            vertices[1, 1] - (vertices[1, 0] - vertices[0, 0]) / (vertices[1, 0] - vertices[2, 0]) * (
                        vertices[1, 1] - vertices[2, 1]))
        # 校正后的四个顶点坐标
        point_set_1 = np.float32([[0, 0], [0, 140], [440, 140], [440, 0]])

    # 校正前平行四边形四个顶点坐标
    new_box = np.array(
        [(vertices[0, 0], vertices[0, 1]), (new_left_point_x, new_left_point_y), (vertices[1, 0], vertices[1, 1]),
         (new_right_point_x, new_right_point_y)])
    point_set_0 = np.float32(new_box)
    return point_set_0, point_set_1, new_box

# 图像透视变换
def transform(img, point_set_0, point_set_1):
    # 变换矩阵
    mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
    # 投影变换
    lic = cv2.warpPerspective(img, mat, (440, 140))
    return lic


# 车牌号码识别
def recognize_license_plate(lic_img):
    # 使用Tesseract进行车牌识别
    gray_lic = cv2.cvtColor(lic_img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    _, thresh_lic = cv2.threshold(gray_lic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
    # 使用pytesseract进行文字识别
    license_number = pytesseract.image_to_string(thresh_lic, config='--psm 8 --oem 3')  # psm=8表示单行文本
    license_number = license_number.strip()

    if len(license_number) > 0:
        license_number = "粤" + license_number[2:]

    # 用单引号将每个字符包裹
    license_number = "'".join(list(license_number))

    # 在每个字符前后加上单引号
    license_number = "'" + license_number + "'"

    return license_number

# 打开文件对话框选择图片
def open_image():
    Tk().withdraw()  # 隐藏主窗口
    file_path = askopenfilename(title="选择车牌图片", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    return file_path


# 在 Tkinter 中展示识别结果
def show_result(license_number, result_image):
    cv2.imshow("License Plate Recognition", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 处理图片并进行车牌识别
def process_image():
    image_path = open_image()
    if not image_path:
        print("未选择文件，程序结束")
        return

    # 图像预处理
    img, img_Gas, img_B, img_G, img_R, img_gray, img_HSV = imgProcess(image_path)

    # 初步识别
    img_bin = preIdentification(img_gray, img_HSV, img_B, img_R)
    points = fixPosition(img, img_bin)
    vertices, rect = findVertices(points)
    point_set_0, point_set_1, new_box = tiltCorrection(vertices, rect)
    lic = transform(img, point_set_0, point_set_1)

    # 识别车牌号码
    license_number = recognize_license_plate(lic)
    print(f"车牌号码是: {license_number}")

    # 创建窗口并显示结果
    final_result = img.copy()

    # 创建一个白色的空白区域来显示车牌识别结果，宽度缩小一半
    result_area_width = 320  # 宽度缩小一半
    result_area = np.zeros((480, result_area_width, 3), dtype=np.uint8)
    result_area.fill(255)  # 填充白色

    # 将车牌图像放大一倍
    lic_resized = cv2.resize(lic, (320, 100))  # 放大车牌图像
    result_area[50:150, 0:320] = lic_resized

    # 使用PIL绘制中文文本
    pil_img = Image.fromarray(result_area)  # 将 NumPy 数组转换为 PIL 图像
    draw = ImageDraw.Draw(pil_img)

    # 你可以使用系统字体文件，或者在此路径下找到支持中文的字体，例如"msyh.ttc"（微软雅黑）
    font = ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", 15)  # 请确保路径正确

    # 绘制文本
    draw.text((10, 20), "车牌定位结果为：", font=font, fill=(0, 0, 0))
    draw.text((10, 170), "车牌的识别结果为：", font=font, fill=(0, 0, 0))
    draw.text((10, 200), f"{license_number}", font=font, fill=(0, 0, 0))
    draw.text((200, 300), "来自摄像头", font=font, fill=(200, 0, 0))
    draw.text((200, 340), "来自照片", font=font, fill=(200, 0, 0))
    # 转回到 NumPy 数组
    result_area = np.array(pil_img)

    # 拼接原图与识别结果
    combined_image = np.hstack((final_result, result_area))  # 将原图和结果拼接在一起

    # 显示合并后的图像
    show_result(license_number, combined_image)

# 创建界面
def create_interface():
    root = Tk()
    root.title("车牌识别系统")

    # 创建一个框架来显示图像和按钮
    frame = Frame(root)
    frame.pack(padx=20, pady=20)

    # 图像展示区域
    label = Label(frame, text="车牌识别系统", font=("Arial", 16))
    label.pack(pady=10)

    # 按钮用来选择并处理图片
    button = Button(frame, text="选择图片并进行车牌识别", command=process_image)
    button.pack(pady=20)

    root.mainloop()


if __name__ == '__main__':
    create_interface()