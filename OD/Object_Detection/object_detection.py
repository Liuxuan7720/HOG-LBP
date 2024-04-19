import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# LBP等价表
lbp_table = [0, 1, 2, 3, 4, 0, 5, 6, 7, 0, 0, 0, 8, 0, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 13, 0, 14, 15, 16,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 19, 0, 20, 21, 22, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 26, 0, 27, 28, 29, 30, 0, 31, 0, 0, 0,
             32, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 36, 37, 0, 38, 0, 0, 0, 39, 0, 0, 0, 0,
             0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 42, 43, 0, 44, 0, 0, 0, 45, 0, 0, 0, 0, 0, 0,
             0, 46, 47, 48, 0, 49, 0, 0, 0, 50, 51, 52, 0, 53, 54, 55, 56, 57]

ori = np.array([[7, 7, 7, 7, 4, 8, 4, 1],
                [7, 7, 7, 8, 4, 4, 1, 1],
                [7, 7, 8, 4, 8, 4, 4, 1],
                [7, 8, 4, 7, 8, 4, 1, 1],
                [7, 8, 7, 7, 4, 8, 1, 1],
                [7, 8, 7, 7, 8, 8, 1, 1],
                [7, 8, 7, 7, 8, 8, 1, 1],
                [7, 7, 7, 8, 8, 1, 1, 1]])

txt_path = "./test_file/direction.txt"
runs_path = "./runs/"


def write_direction_to_txt(direction_data, file_path):
    with open(file_path, 'w') as f:
        for row in direction_data:
            for direction in row:
                f.write(str(direction) + ' ')
            f.write('\n')


# 对图像进行预处理
def pre_process(src, gamma=1.0):
    # 将图像转化为灰度图
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 对图像进行Gamma校正
    inv_gamma = 1 / gamma
    fixed_gray = np.array(np.power((gray / 255), inv_gamma) * 255, dtype=np.uint8)
    # 对图像进行高斯模糊
    # dst = cv2.GaussianBlur(fixed_gray, (3, 3), 1)
    dst = fixed_gray
    # cv2.imshow("pre_process", dst)
    cv2.imwrite(runs_path + "pre_process.png", dst)
    return dst


# 计算图像的梯度的方向与幅值
def gradient(src):
    # 计算x方向和y方向的梯度
    # src=cv2.Canny(src,20,200)
    grad_x = cv2.Sobel(src, -1, 1, 0, ksize=3)
    grad_y = cv2.Sobel(src, -1, 0, 1, ksize=3)
    # 计算梯度的幅值和方向,并将方向转化为角度制，并将其范围限制在0-180
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.degrees(np.arctan2(grad_y, grad_x))
    direction[(grad_x == 0) | (grad_y == 0)] = 500
    return magnitude, direction


# 采用区间方法计算一个Cell内的梯度直方图
def cell_gradient_hist_interval(cell_magnitude, cell_direction, cell_size=8):
    # 初始化直方图
    hist = np.zeros(9)
    # 对像素进行遍历
    for i in range(0, cell_size):
        for j in range(0, cell_size):
            if cell_direction[i][j] != 500 and cell_magnitude[i][j] > 0:
                hist[int(cell_direction[i][j] / 20)] += cell_magnitude[i][j]
    return hist


# 将每个Cell对应于其直方图
def compute_gradient_hist_interval(magnitude, direction, cell_size=8):
    # 获取图像大小
    image_height, image_width = magnitude.shape
    # 计算Cell后的图像大小
    cell_height, cell_width = image_height // cell_size, image_width // cell_size
    # 初始化存储所有区域的直方图
    total_hist = np.zeros((cell_height, cell_width, 9))
    # 遍历整幅图像
    for i in range(cell_height):
        for j in range(cell_width):
            cell_magnitude = magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size: (j + 1) * cell_size]
            cell_direction = direction[i * cell_size:(i + 1) * cell_size, j * cell_size: (j + 1) * cell_size]
            # 计算当前区域的直方图
            total_hist[i, j] = cell_gradient_hist_interval(cell_magnitude, cell_direction, cell_size)
    return total_hist


# 可视化CELL
def visualize_hog(img, gamma=1.0, cell_size=8):
    # 放大图像以增强可视化效果
    img = cv2.resize(img, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # 读取图像大小
    image_height, image_width, channels = img.shape
    # 初始化输出图像
    output_image = np.zeros((image_height, image_width), dtype=np.uint8)
    # 图像预处理
    img = pre_process(img, gamma)
    cv2.imwrite(".runs/hog_pre_process.png", img)
    # 计算梯度
    magnitude, direction = gradient(img)
    # write_direction_to_txt(direction, txt_path)
    # 计算Cell的直方图
    total_hist = compute_gradient_hist_interval(magnitude, direction, cell_size)
    # 遍历图像，绘制归一化的梯度直方图
    # 计算Cell后的图像大小
    cell_height, cell_width = image_height // cell_size, image_width // cell_size
    # 计算有关阈值
    max_norm = np.max(np.linalg.norm(total_hist))
    med_norm = np.median(np.linalg.norm(total_hist))
    # 遍历Cell块
    for i in range(cell_height):
        for j in range(cell_width):
            # 起点坐标
            start_i, start_j = i * cell_size + cell_size // 2, j * cell_size + cell_size // 2
            # 读取梯度并归一化
            hist = total_hist[i, j]
            bright = int(np.min([255, 255 * pow((np.linalg.norm(hist) / med_norm), 0.25)]))
            if np.linalg.norm(hist) > 0:
                hist /= np.linalg.norm(hist)
                # 在当前 Cell 中心处绘制归一化后的梯度
                for angle_bin, normalized_value in enumerate(hist):
                    angle_range = (angle_bin * 20, (angle_bin + 1) * 20)
                    angle = (angle_range[0] + angle_range[1]) / 2
                    angle_rad = np.radians(angle)
                    dx = normalized_value * cell_size * np.sin(angle_rad) * 0.5
                    dy = normalized_value * cell_size * np.cos(angle_rad) * 0.5
                    cv2.line(output_image, (int(start_j - dx), int(start_i - dy)),
                             (int(start_j + dx), int(start_i + dy)), bright, 1)
    cv2.imshow("output", output_image)
    cv2.imwrite(runs_path + "output_hog.png", output_image)
    cv2.waitKey(0)
    return output_image


# 输出Hog特征
def compute_hog_feature(src, gamma=1.0, cell_size=8):
    # 读取图像尺寸并计算Cell的尺寸
    # 读取图像大小
    image_height, image_width, channels = src.shape
    # 初始化输出图形
    output_image = np.zeros((image_height, image_width), dtype=np.uint8)
    # 计算Cell后的图像大小
    cell_height, cell_width = image_height // cell_size, image_width // cell_size
    # 图像预处理
    src = pre_process(src, gamma)
    # 计算梯度
    magnitude, direction = gradient(src)
    # 计算Cell的直方图
    total_hist = compute_gradient_hist_interval(magnitude, direction, cell_size)
    # 计算Hog特征
    # 初始化Hog特征
    hog_feature = []
    # 遍历Cell块儿
    for i in range(cell_height - 1):
        for j in range(cell_width - 1):
            # 初始化block向量
            block_vector = []
            # 将四个cell块儿内的向量放入block内
            # print("total_hist", np.size(total_hist[i, j]))
            block_vector.extend(total_hist[i, j])
            block_vector.extend(total_hist[i + 1, j])
            block_vector.extend(total_hist[i, j + 1])
            block_vector.extend(total_hist[i + 1, j + 1])
            block_vector_form = np.squeeze(np.array(block_vector))
            # print(block_vector_form.shape)
            # 归一化block_vector
            for elements in block_vector:
                elements /= np.linalg.norm(block_vector_form) + 1
            # 将block_vector放入hog_feature中
            hog_feature.extend(block_vector)
    hog_feature_norm = np.squeeze(np.array(hog_feature))
    # print(hog_feature_norm.shape)
    with open("./test_file/hog_feature.txt", 'w') as f:
        for num in hog_feature_norm:
            f.write(str(num) + ' ')
        f.write('\n')
    return hog_feature


# 将binary_value变化为最小的二进制数
def get_min_binary(binary_list, is_min=True):
    s = 0
    l = len(binary_list)
    if not is_min:
        for i in range(l):
            s += int(math.pow(2, i)) * binary_list[l - 1 - i]
    else:
        # 利用双指针求最小二进制数
        left, right = 0, 0
        min_shift, max_zero = 0, 0
        cur_zero = 0
        while left < l and right < l:
            # 若慢指针对应值为1，则清零当前连续0的个数，并向下移位
            if binary_list[left] == 1:
                cur_zero = 0
                left += 1
                right += 1
            else:
                # 若快指针指0，则更新连续0值
                if binary_list[right] == 0:
                    cur_zero += 1
                    right += 1
                    if cur_zero > max_zero:
                        min_shift, max_zero = left, cur_zero
                # 若快指针指1，则将慢指针对齐
                elif binary_list[right] == 1:
                    left = right
        # 计算最小值
        # print((min_shift, max_zero))
        # result = []
        for i in range(l):
            s += int((math.pow(2, i))) * binary_list[(l - 1 - i + min_shift) % l]
        #     result.append(binary_list[(l - 1 - i + min_shift) % l])
        # print(result)
    return s


# 等价类映射
def get_equal_value(num, is_equ):
    if not is_equ:
        return num
    else:
        return lbp_table[num]


# 获取圆形领域
def get_circle_neighbor(points, r):
    neighbor = []
    for i in range(points):
        theta = 2 * i * np.pi / points + np.pi / 2
        dy, dx = round(r * np.sin(theta)), round(r * np.cos(theta))
        neighbor.append([dy, dx])
    return neighbor


# 得到图像对应的LBP矩阵
# 原始矩形定义LBP,type=0时为原始矩形区域，type=1为圆形区域,r为对应半径
def compute_lbp_matrix(src, is_min=True, lbp_type=0, r=1, points=8, is_equ=True):
    # 定义方向数组
    dirs = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    # 读取图像大小
    image_height, image_width, channels = src.shape
    # 图像预处理
    src = pre_process(src)

    # 原始矩形区域
    if lbp_type == 0:
        # 初始化直方图
        lbp_hist = [0] * 256
        # 初始化输出矩阵
        out = np.zeros((image_height, image_width), dtype=np.uint8)
        for i in range(1, image_height - 1):
            for j in range(image_width - 1):
                # 初始化二进制数
                binary_list = [0] * 8
                for k in range(8):
                    dx, dy = dirs[k][0], dirs[k][1]
                    if src[i + dx][j + dy] < src[i][j]:
                        binary_list[k] = 0
                    else:
                        binary_list[k] = 1
                out[i - 1][j - 1] = get_equal_value(get_min_binary(binary_list, is_min), is_equ)
                lbp_hist[out[i - 1][j - 1]] += 1
    # 圆形区域
    else:
        # 初始化输出矩阵
        out = np.zeros((image_height, image_width), dtype=np.uint8)
        # 创建一个圆形区域，仅考虑r<=4的情况
        # 生成对应邻域
        neighbor = get_circle_neighbor(points, r)
        # 初始化直方图
        lbp_hist = [0] * int(math.pow(2, len(neighbor)))
        # 得到对应的二进制数
        for i in range(r, image_height - r):
            for j in range(r, image_width - r):
                # 初始化二进制数
                binary_list = [0] * len(neighbor)
                for k in range(len(neighbor)):
                    dy, dx = neighbor[k][0], neighbor[k][1]
                    if src[i + dy][j + dx] < src[i][j]:
                        binary_list[k] = 0
                    else:
                        binary_list[k] = 1
                        # 去最小二进制数
                out[i - r][j - r] = get_equal_value(get_min_binary(binary_list, is_min), is_equ)
                lbp_hist[out[i - r][j - r]] += 1

    return out, lbp_hist


# 可视化LBP
# type=0为原始矩形区域，type=1为圆形区域，r为圆形区域的半径,is_min表示是否应用旋转不变
def compute_lbp_feature(src, lbp_type=0, is_min=False, r=1, cell_size=16, points=8, is_equ=False):
    # 将图像划分为若干个cell
    image_height, image_width, channels = src.shape
    cell_height, cell_width = image_height // cell_size, image_width // cell_size
    # 初始化输出图像
    if lbp_type == 0:
        output = np.zeros((image_height, image_width), dtype=np.uint8)
        r = 1
    else:
        output = np.zeros((image_height, image_width), dtype=np.uint8)
    # 初始化LBP特征
    lbp_feature = []
    # 遍历所有cell,计算out与hist
    for i in range(cell_height):
        for j in range(cell_width):
            cell = src[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            out, hist = compute_lbp_matrix(cell, is_min, lbp_type, r, points, is_equ)

            # 将hist归一化并保存在指定路径下
            if sum(hist) == 0:
                normalized_hist = hist
            else:
                normalized_hist = [count / sum(hist) for count in hist]
            # 将归一化后的hist存入lbp_feature中
            lbp_feature.extend(normalized_hist)
            # 输出第一个实例直方图
            if i == 0 and j == 0:
                print(normalized_hist)
                plt.bar(range(len(normalized_hist)), normalized_hist)
                plt.xlabel("LBP_Value")
                plt.ylabel("Frequency")
                plt.title("Normalized_Histogram for cell")
                plt.savefig("./runs/lbp_hist/Normalized_Histogram_circle_min_equ" + str(i) + "_" + str(j) + ".png")

            # 将out归一化并映射到output中
            # 归一化
            if np.max(out) == np.min(out):
                normalized_out = out
            else:
                normalized_out = out
            # print(normalized_out)
            # 映射
            # print(i, j)
            output[i * cell_size:(i + 1) * cell_size, j * cell_size: (j + 1) * cell_size] = normalized_out

    cv2.imshow("output_lbp", output)
    cv2.waitKey(0)
    cv2.imwrite("runs/output_lbp_circle_min_equ.png", output)
    print("done!")
    # print(lbp_feature)
    # 将lbp_feature存入指定路径
    with open("./test_file/lbp_feature.txt", 'w') as f:
        for num in lbp_feature:
            f.write(str(num) + ' ')
        f.write('\n')
    return lbp_feature


# 计算图像积分图
def integer_map():
    src = ori
    height, width = src.shape
    result = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            result[i, j] = np.sum(src[0:i+1, 0:j+1])
    print(result)
    return result
