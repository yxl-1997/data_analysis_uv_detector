# -*- coding: utf-8 -*-
# @Time : 2021/4/1 9:13
# @Author : yxl
# @File : UV_experiment.py
# @Project : data_analysis_uv_detector
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import re #正则表达式库
#from matplotlib import rcParams
import matplotlib as mpl
import UV_detectors
#设置打印所有行，所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#打印数组全部元素
np.set_printoptions(threshold=np.inf)
pd.set_option('precision', 10)


'''获得一次实验所有样品编号
参数：
    实验测试数据文件夹编号
返回值：
    文件夹中所有样品所有测试数据编号'''
def get_all_sample_files_df(experiment_dir):
    sample_numbers = os.listdir(experiment_dir)
    #print(sample_numbers)
    files = []
    for i in sample_numbers:
        #print(i)
        file = os.listdir(experiment_dir + '\\' + i)
        files.append(file)
        #print(files)
    files = pd.DataFrame(files)
    files = files.T
    files.columns = sample_numbers #给columns索引从1开始
    all_sample_files_df = files
    print(all_sample_files_df)
    return all_sample_files_df,sample_numbers

def get_wavelength(pc_density_table_name):
    wavelength = pd.read_csv(pc_density_table_name, skiprows=0, usecols=(0,))#只调用一列的时候必须加','
    #print(wavelength)


    return wavelength

'''获得每个样品每个波长下的测试数据
参数：
    传入实验文件夹名称 -> experiment_dir
    文件夹中所有样品所有测试数据编号DataFrame-> all_sample_files_df
返回值：
    所有样品所有测试数据路径'''
def get_IV_data_from_all_sample_files(experiment_dir,sample_numbers,all_sample_files_df):
    #print(all_sample_files_df)
    sample_path = []

    for each_sample in sample_numbers:
        #print(each_sample)
        each_sample_path =experiment_dir + '\\' + '\\'+ str(each_sample)
        sample_path.append(each_sample_path)
    all_sample_path = pd.DataFrame(sample_path)
    all_sample_path = all_sample_path.T
    all_sample_path.columns = sample_numbers


    #print(all_sample_path)
    #print(np.array(all_sample_files_df))
    #for i in all_sample_files_df.items:
    each_file_path =np.array(all_sample_path)  +'\\'+ '\\' + np.array(all_sample_files_df)
    each_file_path = pd.DataFrame(each_file_path)
    each_file_path.columns = sample_numbers
    #print(each_file_path)
    #print(get_wavelength(pc_density_table_name))
    #all_pc_data = []
    #all_dark_data = []
    #print(UV_detectors.data_to_IV('experiment4\\02')[2])#调用第几个返回值
    #for i in all_sample_path.iloc[0]:
        #print(i)
        #all_pc_data.append(UV_detectors.data_to_IV(i)[0])#所有样品，所有波长下光电流数据
        #all_dark_data.append(UV_detectors.data_to_IV(i)[1])
    #all_pc_data= np.array(all_pc_data)
    #all_dark_data = pd.Panel(all_dark_data)
    #print(pd.Panel(all_dark_data))
    #all_net_pc = all_pc_data - all_dark_data
    #print(all_pc_data)
    #print(all_dark_data)

    #print(all_pc_data[:,:,:,1:], all_dark_data.shape)

    return  #all_pc_data,all_dark_data

'''获得所有杨样品0V下响应度
参数：
    样品路径'''

if __name__ == '__main__':
    '''传入实验文件夹名称'''
    experiment_dir = 'experiment4'

    '''传入光功率密度文件名'''
    pc_density_table_name = 'pc_density_table.csv'

    '''获得所有样品编号'''
    all_sample_files_df,sample_numbers = get_all_sample_files_df(experiment_dir)

    '''获得波长测试范围'''
    wavelength = get_wavelength(pc_density_table_name)

    '''获得所有样品数据'''
    IV_data_from_all_sample_files = get_IV_data_from_all_sample_files(experiment_dir,sample_numbers,all_sample_files_df)

