## Hog特征可视化结果
-  
```python
# test visualize
# gamma: 预处理中Gamma校正的参数
# cell_size:每个正方形cell的边长
od.visualize_hog(img, gamma=5.0, cell_size=16)
```
运行结果如下，以下分别为原始图像，预处理后的图像，以及Hog可视化的结果:
<figure>
<img src="E:\RS_EXP2\code\OD\Object_Detection\data\brt.png"\width =120>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\pre_process.png"\width =120>
<img src="E:\RS_EXP2\code\OD\Object_Detection\runs\output_hog.png"\width =120>
</figure>