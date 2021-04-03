# -*- coding: utf-8 -*-
# @Time : 2021/3/27 15:29
# @Author : yxl
# @File : demo1.py
# @Project : data_analysis_uv_detector
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import re #正则表达式库

#打印数组全部元素
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)

'''函数说明：取出样品暗电流装入数组
参数： 
    样品实验数据文件夹名称--dir_name
返回值：
    暗电流I/V矩阵'''
def dark_data_to_IV():

    return

#return dark_I,dark_V

'''函数说明: 将I/V文件中光电流I/V数据提取并装入数组
参数：
    样品实验数据文件夹名--dir_name,电压测试step->step
返回值：
    I/V矩阵'''



def data_to_IV(dir_name):
    files = os.listdir(dir_name)
    #print(files)
    files_csv = list(filter(lambda x : x[-4] == '.csv',files))#找出文件末尾是.csv字符串
    #print(len(files_csv))
    '''暗电流数据获取'''
    dark_file= dir_name + '\\dark.csv'
    dark_data = pd.read_csv(dark_file,skiprows=224,
                            usecols=(1, 2), error_bad_lines=False)

    #print("这是暗电流数据:\n",dark_data)
    dark_data_array = pd.DataFrame(dark_data)
    print(dark_data_array)
    '''光电流数据获取:确定文件夹最后一行数据为暗电流数据'''
    light_files = files[:len(files)-1]
    #print(light_datas)
    pc_data_array = []
    for light_file_names in light_files:
        light_file_names = dir_name + '\\' + light_file_names
        tmp = pd.read_csv(light_file_names,
                            skiprows=224, usecols=(1, 2), error_bad_lines=False)
        #print(tmp.head(11))
        pc_data_array.append(tmp)
    #print('********这是光电流数据:**********\n',pc_data_array)
    return pc_data_array,dark_data_array

'''函数说明：取正向或者负向电流
参数：pc_data_array
返回值：正向电压电流数据 -> pc_positive_value_array或者反向电压电流数据 -> pc_negative_value_array'''
def pc_positive_or_negative_value(pc_data_array):
    pc_positive_value_array = []
    pc_negative_value_array = []
    for each_data in pc_data_array:
        pc_positive_value = each_data[len(each_data)//2:]
        pc_positive_value.append(pc_positive_value)
        pc_negative_value = each_data[:len(each_data)//2+1]
        pc_negative_value_array.append(pc_negative_value)
        #print(pc_positive_value)
    #print(pc_negative_value_array,pc_negative_value_array)
    #print(len(pc_data_array))
    return pc_positive_value_array,pc_negative_value_array
        #找出所有光电流数据
            # light_data = pd.read_csv(light_file,
            #                 skiprows=224, usecols=(1, 2), error_bad_lines=False)
            # light_data_array.append(light_data)

        #return light_datas,dark_data

'''函数说明：取出210nm光电流值
参数：
    光电流数据->light_data,需要绘制的光电流波长->light_wavelength
返回值：
    210nm光电流
'''
#def get_210nm_light_data(light_datas,light_wavelength):
   # for light_plot in light_datas:
        #if light_plot == light_wavelength:

    #return light_plot



'''函数说明：绘制暗电流、210nm光电流曲线
横轴电压V，纵轴电流I'''
def draw_light_dark_IV_curve(dark_data_array):

    print(X_value)
    # plt.figure()
    # plt.scatter(X_value[:,0],X_value[:,1],s=50, cmap='viridis')
    # plt.title('Dataset')
    # plt.show()







#get_210nm_light_data(light_datas=)

if __name__ == '__main__':
    dir_name = 'experiment4\\02'
    pc_data_array,dark_data_array= data_to_IV(dir_name)
    draw_light_dark_IV_curve(dark_data_array)
