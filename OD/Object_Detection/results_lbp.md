## LBP特征可视化结果
- 
#### 1. 原始矩形LBP特征
##### 1.1 无旋转不变与等价类
参数选择如下：
```python
# test compute_lbp_matrix

# lbp_type： lbp的邻域类型，lbp_type=0时为原始矩形区域，lbp_type=1时为圆形区域
# r: 当lbp_type=1时，圆形邻域的半径
# points: 当lbp_type=1时，圆形邻域的采样点个数
# is_min: 是否取循环最小二进制数
# is_equ: 是否取等价类

od.compute_lbp_feature(img, cell_size=512, r=2, lbp_type=0, points=12, is_min=False, is_equ=False)
```
以下分别为原始图像，LBP特征可视化即LBP值的分布直方图

<figure>
<img src="E:\RS_EXP2\code\OD\Object_Detection\data\lena.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\output_lbp_origin_square.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\lbp_hist\Normalized_Histogram_square0_0.png"\width =147>
</figure>

##### 1.2 含有旋转不变，不含等价类
参数选择如下:
```python
# test compute_lbp_matrix
od.compute_lbp_feature(img, cell_size=512, r=2, lbp_type=0, points=12, is_min=True, is_equ=False)
```
<figure>
<img src="E:\RS_EXP2\code\OD\Object_Detection\data\lena.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\output_lbp_square_min.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\lbp_hist\Normalized_Histogram_square_min0_0.png"\width =147>
</figure>

##### 1.3 旋转不变等价类
参数选择如下：
```python
# test compute_lbp_matrix
od.compute_lbp_feature(img, cell_size=512, r=2, lbp_type=0, points=12, is_min=True, is_equ=True)
```
<figure>
<img src="E:\RS_EXP2\code\OD\Object_Detection\data\lena.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\output_lbp_square_min_equ.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\lbp_hist\Normalized_Histogram_square_min_equ0_0.png"\width =147>
</figure>

#### 2.圆形LBP特征
#### 2.1 无旋转不变，无等价类
参数选择如下：
```python
# test compute_lbp_matrix
od.compute_lbp_feature(img, cell_size=512, r=2, lbp_type=1, points=12, is_min=False, is_equ=False)
```
<figure>
<img src="E:\RS_EXP2\code\OD\Object_Detection\data\lena.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\output_lbp_circle.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\lbp_hist\Normalized_Histogram_circle0_0.png"\width =147>
</figure>

#### 2.2 旋转不变，无等价类
参数选择如下：
```python
# test compute_lbp_matrix
od.compute_lbp_feature(img, cell_size=512, r=2, lbp_type=1, points=12, is_min=True, is_equ=False)
```
<figure>
<img src="E:\RS_EXP2\code\OD\Object_Detection\data\lena.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\output_lbp_circle.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\lbp_hist\Normalized_Histogram_circle_min0_0.png"\width =147>
</figure>

#### 2.3 旋转不变等价类
参数选择如下：
```python
# test compute_lbp_matrix
od.compute_lbp_feature(img, cell_size=512, r=2, lbp_type=1, points=12, is_min=True, is_equ=True)
```
<figure>
<img src="E:\RS_EXP2\code\OD\Object_Detection\data\lena.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\output_lbp_circle_min_equ.png"\width =110>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\lbp_hist\Normalized_Histogram_circle_min_equ0_0.png"\width =147>
</figure>
