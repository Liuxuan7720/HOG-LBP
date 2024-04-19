import object_detection as od
import numpy as np
import cv2

# read image
img_path = "./data/brt.png"
img = cv2.imread(img_path)
# cv2.imshow("lena", img)
# cv2.waitKey(0)

# 需要分别测试
# test pre_progress
img = od.pre_process(img, 1)
cv2.imshow("lena", img)
cv2.waitKey(0)

# test gradient
# img = od.pre_process(img, 1)
# magnitude, direction = od.gradient(img)
# print(magnitude)
# print(direction)


# # test visualize
od.visualize_hog(img, gamma=5.0, cell_size=16)

# test hog_feature
# od.compute_hog_feature(img, gamma=1.0, cell_size=8)

# test get_min_binary
# value = [1, 0, 0, 0, 1, 0, 1, 1]
# print(od.get_min_binary(value, is_min=True))

# # test compute_lbp_matrix
od.compute_lbp_feature(img, cell_size=512, r=2, lbp_type=1, points=8, is_min=True, is_equ=True)

# test integer_map
od.integer_map()