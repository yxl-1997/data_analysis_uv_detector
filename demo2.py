# -*- coding: utf-8 -*-
# @Time : 2021/3/29 21:24
# @Author : yxl
# @File : demo2.py
# @Project : data_analysis_uv_detector
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
x=[]
Y=[]
for i in range(3):
    x.append(i)
    y=np.random.random()
    Y.append(y)
plt.scatter(x,Y)
plt.show()