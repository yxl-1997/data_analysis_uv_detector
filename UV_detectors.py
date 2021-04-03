# -*- coding: utf-8 -*-
# @Time : 2021/3/29 9:02
# @Author : yxl
# @File : UV_detectors.py
# @Project : data_analysis_uv_detector

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import re #正则表达式库
#from matplotlib import rcParams
import matplotlib as mpl

#设置打印所有行，所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#打印数组全部元素
np.set_printoptions(threshold=np.inf)
pd.set_option('precision', 10)
'''函数说明: 将I/V文件中光电流I/V数据提取并装入数组
参数：
    样品实验数据文件夹名--dir_name,电压测试step->step
返回值：
    光电流I-V DataFrame ->pc_data_array
    暗电流DataFrame -> dark_data_array
    样品编号 -> samople_number
    I/V矩阵'''
def data_to_IV(dir_name):
    files = os.listdir(dir_name)
    #print(files)
    files_csv = list(filter(lambda x : x[-4] == '.csv',files))#找出文件末尾是.csv字符串
    #print(len(files))
    '''暗电流数据获取'''
    dark_file= dir_name + '\\dark.csv'
    #print(dark_file)
    sample_number = 'sample ' + dir_name[-1]
    #print(sample_number)
    dark_data = pd.read_csv(dark_file,skiprows=224,
                            usecols=(1, 2), error_bad_lines=False)

    #print("这是暗电流数据:\n",dark_data)
    dark_data_array = pd.DataFrame(dark_data)
    #print(dark_data_array)
    '''光电流数据获取:确定文件夹最后一行数据为暗电流数据'''
    light_files = files[:len(files)-1]
    #print(light_files)
    #print(light_datas)
    pc_data_array = []
    for light_file_names in light_files:
        light_file_names = dir_name + '\\' + light_file_names
        tmp = pd.read_csv(light_file_names,
                            skiprows=224, usecols=(1, 2), error_bad_lines=False)
        #print(tmp.head(11))
        pc_data_array.append(tmp)

    #print('********这是暗电流数据:**********\n',dark_data_array,dark_data_array.shape)
    #print('********这是光电流数据:**********\n',pc_data_array,pc_data_array.__len__())

    return pc_data_array,dark_data_array,sample_number

'''函数说明：获得探测器测试电压值
参数：
    暗电流数组 -> dark_data_array
返回值：
    探测器电压数组 -> volts_array'''
def get_volts_as_index_array(dark_data_array):
    #print(dark_data_array)
    volts_array = dark_data_array[' V1'].round(1)# round(1) ->只保留一位小数
    #print(volts_array)
    #volts_as_index = pd.DataFrame({'V':[volts_array]})
    volts_as_index = pd.DataFrame(volts_array).set_index(' V1').astype(float).T
    #print(volts_as_index.dtypes)

    #print(volts_as_index.head(10))
    return volts_as_index

''''函数说明：获取紫外波段净光电流（光电流-暗电流）数据并装入数组  
参数：
    所有波长下光电流DataFrame -> pc_data_array
    暗电流数组-> dark_data_array
返回值：
    净光电流数组 DataFrame -> net_pc_current'''
def get_net_pc_data(pc_data_array,dark_data_array):
    #print(pc_data_array)
    net_pc_data_array = pd.DataFrame()
    for each_pc_data in pc_data_array:
        #print(each_pc_data)
        net_pc_data =  each_pc_data[' I1'] - dark_data_array[' I1'] #所有电压下净光电流
        net_pc_data_array = net_pc_data_array.append(net_pc_data)

    #print('********这是净光电流电流数据:**********\n',net_pc_data_array,net_pc_data_array.shape)

    return net_pc_data_array

'''函数说明：找出特定电压值对应的净光带电流中的列索引
参数：
    需要的电压列表 -> volts_list
    电压间隔 -> volt_step
返回值：
    特定电压对应的净光电流的列索引'''
def get_net_pc_columns_index(volts_list,volt_step):
    net_pc_columns_index = []
    for i in volts_list:
        tmp = i * 10 * volt_step
        net_pc_columns_index.append(tmp)
    #print(net_pc_columns_index)
    return net_pc_columns_index

'''函数说明：提取光功率密度表中光功率密度所在列  
参数：
    光功率密度表文件名 -> pc_density_table_name
返回值：
    光功率密度DataFrame -> pc_density'''
def get_pc_density(pc_density_table_name):

    pc_density_data  = pd.read_csv(pc_density_table_name, skiprows=0, usecols=(0, 4))
    pc_density = pc_density_data#["光功率密度W/mm2"]
    pc_density_without_wavelength = np.squeeze(pc_density_data["光功率密度W/mm2"],axis=0)
    wavelength = pc_density_data["波长"]

    #print('这是光功率密度：\n',pc_density)
    #print('这是光功率密度(不含波长)：\n',pc_density_without_wavelength,pc_density_without_wavelength.shape)
    #print(wavelength)
    return pc_density,pc_density_without_wavelength,wavelength

'''函数说明：计算探测器上光功率（W）： 探测器上光功率 = 光功率密度*有效光敏面积
参数：
    电极有效面积 effect_area  -> A
    光功率密度 -> pc_density
返回值：
    探测器上光功率detector_acquired_optical_power'''
def get_detector_acquired_optical_power(effect_area,pc_density_without_wavelength):
    #print(pc_density_without_wavelength)

    detector_acquired_optical_power =pd.DataFrame( effect_area * pc_density_without_wavelength)
    detector_acquired_optical_power = detector_acquired_optical_power.values#只取value值 返回的是'numpy.ndarray'
    #print('********这是探测器上光功率:**********\n',detector_acquired_optical_power)
    return detector_acquired_optical_power

'''函数说明：获得特定电压值对应的净光带电流中的列索引所对应的净电流值   
参数：
    净光电流数组 -> net_pc_data_array
    特定电压下数组索引 -> net_pc_columns_index
    探测器测试电压作为索引 -> volts_as_index
返回值：
    特定电压值对应的净光带电流中的列索引 -> net_pc_columns_index'''
def get_net_pc_at_volts(net_pc_data_array,
                                net_pc_columns_index,volts_as_index):
    volts_as_index_list = []
    for i in volts_as_index:
        volts_as_index_list .append(i)
    #print(volts_as_index_list)

    #print(net_pc_columns_index)
    #net_pc_data_array = pd.DataFrame(net_pc_data_array)
    net_pc_data_array.columns = volts_as_index_list #替换净光电流columns索引值
    #net_pc_data_array.rename(columns=volts_as_index_list)
    #print('********这是净光电流tali(5):**********\n',net_pc_data_array.tail(5))
    #print(net_pc_data_array)
    #print(net_pc_data_array)
    net_pc_at_volts = []
    for i in net_pc_columns_index:
         #print(i)
         net_pc_at_volts.append(net_pc_data_array[i])

    #print('********这是你所需要的特定电压下的净光电流:**********\n',net_pc_columns_index ,'V','\n',net_pc_at_volts)
    return net_pc_at_volts

'''计算PDCR Photo to dark current ratio
参数：
    光电流数据 -> pc_data_array
    暗电流数据 -> dark_data_array
返回值：
    PDCR'''
def calculate_PDCR_df_at_Volts_and_pc_density(pc_data_array,dark_data_array,volts_list,pc_density_without_wavelength):
    #print(wavelength)
    #print(dark_data_array)
    dark_data_array_only_current = np.array(dark_data_array)[:,1:]
    pc_data_array_only_current = np.array(pc_data_array)[:,:,1:]
    #print(dark_data_array_only_current.shape,pc_data_array_only_current.shape)
    PDCR = (pc_data_array_only_current - dark_data_array_only_current) / dark_data_array_only_current
    PDCR_df = pd.DataFrame(np.squeeze(PDCR,axis=2))
    PDCR_df.columns = dark_data_array[' V1']
    PDCR_df.set_index(pc_density_without_wavelength,inplace=True)
    PDCR_df_at_Volts_and_pc_density = PDCR_df[volts_list]
    #print('这是特定电压下PDCR随光功率密度变化\n',PDCR_df_at_Volts_and_pc_density)
    return PDCR_df_at_Volts_and_pc_density


'''函数说明：计算探测器响应度
参数：
    特定电压下净光电流 -> net_pc_at_volts   Responsivity = (特定电压下净光电流/探测器上光功率)
    探测器上光功率 -> detector_acquired_optical_power
返回值：
    探测器响应度 -> Responsivity'''
def calculate_Responsivity(net_pc_at_volts,detector_acquired_optical_power,wavelength,volts_list):
    net_pc_at_volts = np.array(net_pc_at_volts)#将列表转成array
    net_pc_at_volts = pd.DataFrame(net_pc_at_volts)#将array转成DataFrame格式
    net_pc_at_volts.columns = wavelength#将行索引改成波长
    net_pc_at_volts = net_pc_at_volts.T#将DataFrame转置
    Responsivity = net_pc_at_volts / detector_acquired_optical_power
    Responsivity.columns = volts_list
    #print('********这是特定电压下的响应度:**********\n',Responsivity,type(Responsivity))

    return Responsivity

'''计算所有样品0V下响应度
参数：
    所有样品路径
返回值：所有样品0V下响应度
'''
# #def calculate_Responsivityat_0V_of_all_sample(dir_name):
#     #print(dir_name)
#     experiment_dir = dir_name[0:11]
#     #print(experiment_dir)
#     sample_number = os.listdir(experiment_dir)
#     all_data = []
#     for i in sample_number:
#         sample_path = experiment_dir + '\\' + i
#         all_data.append(data_to_IV(sample_path))
#     all_data = np.squeeze(all_data)
#
#
#
#
#
#
#     return

'''计算特定电压下紫外/可见光抑制比
计算公式：Rejection_ratio = R_210/R_400
参数：
    特定电压下响应度 -> Responsivity
    
返回值：
    特定电压下紫外/可见光抑制比 -> Rejection_ratio 
'''
def calculate_Rejection_ratio(Responsivity,UV_wavelength):
    Responsivity = pd.DataFrame(Responsivity)
    #print(Responsivity)
    UV_wavelength = int(UV_wavelength)
    #print(type(UV_wavelength))
    #print(type(Responsivity.index))
    # for i in Responsivity.index:
    #     if i == UV_wavelength:
    #         #print(Responsivity.loc[UV_wavelength])
    Responsivity_at_wavelength = Responsivity.loc[UV_wavelength]
    #print(Responsivity_at_wavelength,Responsivity.loc[400])
    Rejection_ratio = Responsivity_at_wavelength / Responsivity.loc[400]

    #print(Rejection_ratio)

    return Rejection_ratio

'''计算探测度Detectivity
    计算公式：Detectivity = 
参数：
    暗电流 -> dark_data_array
    特定电压下响应度 -> Responsivity
    特定电压列表 -> volts_list
'''
def cauculate_Detectivity(dark_data_array,Responsivity,volts_list,UV_wavelength,effective_area):
    dark_data_array.set_index(' V1',inplace=True )
    dark_I_at_volts =dark_data_array.loc[volts_list].T
    #print(type(dark_I_at_volts))

    #print(Responsivity)
    #dark_data_at_volts_density = dark_data_array.loc[volts_list] / effect_area
    q = math.pow(10,-19) * 1.6

   # print(math.sqrt(dark_I_at_volts.values * 2 * q))

    #fenzi = Responsivity * (math.sqrt(effective_area))

    #print(volts_list)
    #print(dark_data_at_volts_density.shape)
    #print(np.array(Responsivity.loc[UV_wavelength]).shape)


    Detectivity = Responsivity.values * (math.sqrt(effective_area))/((dark_I_at_volts.values * 2 * q)**(1/2))
    #print(math.sqrt((2*q*dark_I_at_volts)))
        #Responsivity.loc[UV_wavelength] / dark_data_at_volts_density
    Detectivity = pd.DataFrame(Detectivity)
    Detectivity.set_index(wavelength,inplace=True)

    #print(Detectivity,Detectivity.shape)
    return Detectivity

'''函数说明：取正向或者负向电流
参数：pc_data_array
返回值：正向电压电流数据 -> pc_positive_value_array
或者反向电压电流数据 -> pc_negative_value_array
'''
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

'''函数说明：获取紫外波段光电流数据并装入数组
参数：
    文件夹名称 -> dir_name
    紫外波长-> pc_wavelength   example:210nm
返回值：
    特定波长下光电流数组 DataFrame'''
def get_pc_current(dir_name,pc_wavelength):
    pc_current_file = dir_name + '\\'+ pc_wavelength +'.csv'  #光电流文件
    pc_current_data = pd.read_csv(pc_current_file, skiprows=224,
                            usecols=(1, 2), error_bad_lines=False)

    #print(pc_current_data)
    return pc_current_data,pc_wavelength

'''函数说明：绘制暗电流、210nm光电流曲线 -> 横轴电压V，纵轴电流I
参数：
    暗电流数组DataFrame -> dark_data_array
    光电流数组DataFrame -> pc_current_data
    光电流波长 -> pc_curretn 
    样品编号 -> sample_number
返回值：
    暗电流与特定波长下暗/光电流曲线'''
def draw_light_dark_IV_curve(dark_data_array,pc_current_data,pc_wavelength,sample_number):
    #print(dark_data_array)
    X_V = dark_data_array[' V1']  # 按照列名索引 '''注意列索引前面有一个空格'''
    #print(X_V)
    Y_dark_I = dark_data_array[' I1']  # 按照列名索引 '''注意列索引前面有一个空格'''
    Y_uv_I = pc_current_data[' I1']
    X_range = np.arange(min(X_V), max(X_V) + 1, 1)  # 设置横轴图像刻度

    # 设置西文字体为新罗马字体
    fig = plt.figure(dpi=200)
    figure, ax = plt.subplots(figsize=(12, 9))#创建一个画板
    config = {
        "font.family": 'Times New Roman',   # 设置字体类型
        "font.size": 12,                    # 设置字体大小
        "font.weight": 'bold',              # 设置字体加粗

        "mathtext.fontset":'Times New Roman',
    }
    title_text = sample_number
    # 设置横轴范围
    plt.scatter(X_V,Y_dark_I,color = 'black',s=100, cmap='viridis',label='dark',linewidths=3)
    plt.scatter(X_V,Y_uv_I,color = 'b',s=100, cmap='viridis',marker='*',label=pc_wavelength + 'nm',linewidths=3 )
    #设置标题
    plt.title(sample_number, fontdict={'family': 'Times New Roman', 'size': 30, 'weight': 'bold'})#设置标题
    # 设置刻度
    plt.yticks(fontproperties='Times New Roman', weight='bold', size=30)
    plt.xticks(X_range,fontproperties='Times New Roman', weight='bold', size=30)#设置横轴图像刻度
    #设置横纵轴说明
    plt.xlabel("Bias/V",fontproperties='Times New Roman', weight='bold', size=30)#设置X轴图示，字体，加粗
    plt.ylabel("Current/A",fontproperties='Times New Roman', weight='bold', size=30)
    #设置图例
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 30,
             }
    plt.legend(loc='lower right',frameon=False,fontsize=30,prop = font1 )#设置图例在右下角显示，取消边框
    #设置线宽
    ax_width = 3.5
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    # 设置刻度朝向
    ax.tick_params(axis='both', which='both', direction='in',size = 5,colors = 'black',width = 3)
    '''matplotlib.pyplot.tick_params参数:
    axis : 可选{‘x’, ‘y’, ‘both’} ，选择对哪个轴操作，默认是’both’
    reset : bool，如果为True，则在处理其他参数之前将所有参数设置为默认值。 它的默认值为False。
    which : 可选{‘major’, ‘minor’, ‘both’} 选择对主or副坐标轴进行操作
    direction/tickdir : 可选{‘in’, ‘out’, ‘inout’}刻度线的方向
    size/length : float, 刻度线的长度
    width : float, 刻度线的宽度
    color : 刻度线的颜色，我一般用16进制字符串表示，eg：’#EE6363’
    pad : float, 刻度线与刻度值之间的距离
    labelsize : float/str, 刻度值字体大小
    labelcolor : 刻度值颜色
    colors : 同时设置刻度线和刻度值的颜色
    zorder : float ，Tick and label zorder.
    bottom, top, left, right : bool, 分别表示上下左右四边，是否显示刻度线，True为显示
    labelbottom, labeltop, labelleft, labelright :bool, 分别表示上下左右四边，是否显示刻度值，True为显示
    labelrotation : 刻度值逆时针旋转给定的度数，如20
    gridOn: bool ,是否添加网格线； grid_alpha:float网格线透明度 ； grid_color: 网格线颜色; grid_linewidth:float网格线宽度； grid_linestyle: 网格线型
    tick1On, tick2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度线
    label1On,label2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度值'''
    plt.style.use('ggplot')
    #print("*"*10+'这是电压值'+"*"*10+'\n',"\n".join(str(x) for x in X_V.values))
    #print("*"*10+'这是暗电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_dark_I.values))
    #print("*"*10+'这是光电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_uv_I.values))
    plt.show()
    volt_dark_pc = pd.concat([X_V, Y_dark_I,Y_uv_I], axis=1)
    print('*********这是电压、暗电流'+pc_wavelength+'nm光电流数据***********\n',volt_dark_pc)
    #print('这是电压值\n',X_V.values,'\n','这是暗电流值\n',Y_dark_I.values,'\n','这是' + pc_wavelength + '电压值\n',Y_uv_I.values)

    return volt_dark_pc

'''函数说明：绘制暗电流、210nm光电流曲线 -> 横轴电压V，纵轴电流I
参数：
    暗电流数组DataFrame -> dark_data_array
    光电流数组DataFrame -> pc_current_data
    光电流波长 -> pc_curretn 
    样品编号 -> sample_number
返回值：
    暗电流与特定波长下半对数暗/光电流曲线'''
def draw_semilog_light_dark_curve(dark_data_array,pc_current_data,pc_wavelength,sample_number):
    X_Values = dark_data_array[' V1']  # 按照列名索引 '''注意列索引前面有一个空格'''
    Y_dark_I = dark_data_array[' I1']  # 按照列名索引 '''注意列索引前面有一个空格'''
    Y_uv_I = pc_current_data[' I1']
    X_range = np.arange(min(X_Values), max(X_Values) + 1, 1)  # 设置横轴图像刻度

    # 设置西文字体为新罗马字体
    #fig = plt.figure(dpi=200)
    figure, ax = plt.subplots(figsize=(12, 9))  # 创建一个画板
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": 12,  # 设置字体大小
        "font.weight": 'bold',  # 设置字体加粗

        "mathtext.fontset": 'Times New Roman',
    }
    title_text = sample_number
    plt.scatter(X_Values,abs(Y_dark_I), color='black', s=100, cmap='viridis', label='dark', linewidths=3)
    plt.scatter(X_Values,abs(Y_uv_I), color='b', s=100, cmap='viridis', marker='*', label=pc_wavelength + 'nm', linewidths=3)
    # 设置标题
    plt.title(sample_number, fontdict={'family': 'Times New Roman', 'size': 30, 'weight': 'bold'})  # 设置标题
    # 设置刻度
    plt.yticks(fontproperties='Times New Roman', weight='bold', size=30)
    plt.xticks(X_range, fontproperties='Times New Roman', weight='bold', size=30)  # 设置横轴图像刻度
    # 设置横纵轴说明
    plt.xlabel("Bias/V", fontproperties='Times New Roman', weight='bold', size=30)  # 设置X轴图示，字体，加粗
    plt.ylabel("Current/A", fontproperties='Times New Roman', weight='bold', size=30)
    # 设置图例
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 30,
             }
    plt.legend(loc='lower right', frameon=False, fontsize=30, prop=font1)  # 设置图例在右下角显示，取消边框
    # 设置线宽
    ax_width = 3.5
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    # 设置刻度朝向
    ax.tick_params(axis='both', which='both', direction='in', size=5, colors='black', width=3)

    ax.set_yscale("log")
    '''matplotlib.pyplot.tick_params参数:
    axis : 可选{‘x’, ‘y’, ‘both’} ，选择对哪个轴操作，默认是’both’
    reset : bool，如果为True，则在处理其他参数之前将所有参数设置为默认值。 它的默认值为False。
    which : 可选{‘major’, ‘minor’, ‘both’} 选择对主or副坐标轴进行操作
    direction/tickdir : 可选{‘in’, ‘out’, ‘inout’}刻度线的方向
    size/length : float, 刻度线的长度
    width : float, 刻度线的宽度
    color : 刻度线的颜色，我一般用16进制字符串表示，eg：’#EE6363’
    pad : float, 刻度线与刻度值之间的距离
    labelsize : float/str, 刻度值字体大小
    labelcolor : 刻度值颜色
    colors : 同时设置刻度线和刻度值的颜色
    zorder : float ，Tick and label zorder.
    bottom, top, left, right : bool, 分别表示上下左右四边，是否显示刻度线，True为显示
    labelbottom, labeltop, labelleft, labelright :bool, 分别表示上下左右四边，是否显示刻度值，True为显示
    labelrotation : 刻度值逆时针旋转给定的度数，如20
    gridOn: bool ,是否添加网格线； grid_alpha:float网格线透明度 ； grid_color: 网格线颜色; grid_linewidth:float网格线宽度； grid_linestyle: 网格线型
    tick1On, tick2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度线
    label1On,label2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度值'''
    plt.style.use('ggplot')
    # print("*"*10+'这是电压值'+"*"*10+'\n',"\n".join(str(x) for x in X_V.values))
    # print("*"*10+'这是暗电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_dark_I.values))
    # print("*"*10+'这是光电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_uv_I.values))
    plt.show()
    semilog_volt_dark_pc = pd.concat([X_Values, abs(Y_dark_I), abs(Y_uv_I)], axis=1)
    print('*********这是电压、暗电流' + pc_wavelength + 'nm光电流半对数数据(对电流值取绝对值)***********\n', semilog_volt_dark_pc)
    return None

'''绘制特定电压下响应度曲线
参数：
    计算的得到的特定电压下响应度数据 -> Responsivity
    需要的电压列表 -> volts_list
    样品编号 -> sample_number
返回值：
    None'''
def plot_Responsivity_curve(Responsivity,volts_list,wavelength,sample_number):
    figure, ax = plt.subplots(figsize=(12, 9))  # 创建一个画板
    X_Values = wavelength.values
    # Y_Values = []
    X_range = np.arange(min(wavelength.values), max(wavelength.values), 50)
    #print(X_range)
    for i in volts_list:
        #     Y_Values.append(Responsivity[i])i
        #     Y_Values.append(Responsivity[i])
        plt.plot(X_Values, abs(Responsivity[i]),
                 label=(format(str(i) + "V")), linewidth=10)
    # ax.cla()
    # plt.pause(0.1)

    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": 12,  # 设置字体大小
        "font.weight": 'bold',  # 设置字体加粗

        "mathtext.fontset": 'Times New Roman',
    }
    title_text = sample_number
    # plt.scatter(X_Values, Responsivity[], color='black', s=100, cmap='viridis', label='dark', linewidths=3)
    # plt.scatter(X_Values, Responsivity[1], color='b', s=100, cmap='viridis', marker='*', label=pc_wavelength + 'nm',linewidths=3)
    # 设置标题
    plt.title(title_text, fontdict={'family': 'Times New Roman', 'size': 30, 'weight': 'bold'})  # 设置标题
    # 设置刻度
    plt.yticks(fontproperties='Times New Roman', weight='bold', size=30)
    plt.xticks(X_range, fontproperties='Times New Roman', weight='bold', size=30)  # 设置横轴图像刻度
    # 设置横纵轴说明
    plt.xlabel("Wavelength/(λ)", fontproperties='Times New Roman', weight='bold', size=30)  # 设置X轴图示，字体，加粗
    plt.ylabel("Responsivity/(A/W)", fontproperties='Times New Roman', weight='bold', size=30)
    # 设置图例

    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 20,
             }
    plt.legend(loc='upper right', frameon=False, fontsize=20, prop=font1)  # 设置图例在右下角显示，取消边框
    # 设置线宽
    ax_width = 3.5
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    # 设置刻度朝向
    ax.tick_params(axis='both', which='both', direction='in', size=5, colors='black', width=3,
                   pad=15)  # pad是调整刻度线与刻度之间的距离

    #ax.set_yscale("log")

    # plt.style.use('ggplot')
    # print("*"*10+'这是电压值'+"*"*10+'\n',"\n".join(str(x) for x in X_V.values))
    # print("*"*10+'这是暗电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_dark_I.values))
    # print("*"*10+'这是光电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_uv_I.values))
    plt.show()

    return None

'''绘制半对数坐标响应度曲线

'''
def plot_semilog_Responsivity_curve(Responsivity,volts_list,wavelength,sample_number):

    figure, ax = plt.subplots(figsize=(12, 9))  # 创建一个画板
    X_Values = wavelength.values
    #Y_Values = []
    X_range = np.arange(min(wavelength.values), max(wavelength.values), 50)
    #print(X_range)
    for i in volts_list:
    #     Y_Values.append(Responsivity[i])i
    #     Y_Values.append(Responsivity[i])
        plt.plot(X_Values, abs(Responsivity[i]),
                     label=(format(str(i)+"V")),linewidth =10)
    #ax.cla()
    #plt.pause(0.1)



    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": 12,  # 设置字体大小
        "font.weight": 'bold',  # 设置字体加粗

        "mathtext.fontset": 'Times New Roman',
    }
    title_text = sample_number
    #plt.scatter(X_Values, Responsivity[], color='black', s=100, cmap='viridis', label='dark', linewidths=3)
    #plt.scatter(X_Values, Responsivity[1], color='b', s=100, cmap='viridis', marker='*', label=pc_wavelength + 'nm',linewidths=3)
    # 设置标题
    plt.title(title_text, fontdict={'family': 'Times New Roman', 'size': 30, 'weight': 'bold'})  # 设置标题
    # 设置刻度
    plt.yticks(fontproperties='Times New Roman', weight='bold', size=30)
    plt.xticks(X_range, fontproperties='Times New Roman', weight='bold', size=30)  # 设置横轴图像刻度
    # 设置横纵轴说明
    plt.xlabel("Wavelength/(λ)", fontproperties='Times New Roman', weight='bold', size=30)  # 设置X轴图示，字体，加粗
    plt.ylabel("Responsivity/(A/W)", fontproperties='Times New Roman', weight='bold', size=30)
    # 设置图例

    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 20,
             }
    plt.legend(loc='upper right', frameon=False, fontsize=20,prop = font1)  # 设置图例在右下角显示，取消边框
    # 设置线宽
    ax_width = 3.5
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    # 设置刻度朝向
    ax.tick_params(axis='both', which='both', direction='in', size=5, colors='black', width=3,pad=15)#pad是调整刻度线与刻度之间的距离

    ax.set_yscale("log")
    '''matplotlib.pyplot.tick_params参数:
    axis : 可选{‘x’, ‘y’, ‘both’} ，选择对哪个轴操作，默认是’both’
    reset : bool，如果为True，则在处理其他参数之前将所有参数设置为默认值。 它的默认值为False。
    which : 可选{‘major’, ‘minor’, ‘both’} 选择对主or副坐标轴进行操作
    direction/tickdir : 可选{‘in’, ‘out’, ‘inout’}刻度线的方向
    size/length : float, 刻度线的长度
    width : float, 刻度线的宽度
    color : 刻度线的颜色，我一般用16进制字符串表示，eg：’#EE6363’
    pad : float, 刻度线与刻度值之间的距离
    labelsize : float/str, 刻度值字体大小
    labelcolor : 刻度值颜色
    colors : 同时设置刻度线和刻度值的颜色
    zorder : float ，Tick and label zorder.
    bottom, top, left, right : bool, 分别表示上下左右四边，是否显示刻度线，True为显示
    labelbottom, labeltop, labelleft, labelright :bool, 分别表示上下左右四边，是否显示刻度值，True为显示
    labelrotation : 刻度值逆时针旋转给定的度数，如20
    gridOn: bool ,是否添加网格线； grid_alpha:float网格线透明度 ； grid_color: 网格线颜色; grid_linewidth:float网格线宽度； grid_linestyle: 网格线型
    tick1On, tick2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度线
    label1On,label2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度值'''
    #plt.style.use('ggplot')
    # print("*"*10+'这是电压值'+"*"*10+'\n',"\n".join(str(x) for x in X_V.values))
    # print("*"*10+'这是暗电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_dark_I.values))
    # print("*"*10+'这是光电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_uv_I.values))
    plt.show()


    return None

'''绘制特定电压下归一化响应度曲线

   公式：X_norm= (X - X_min) / X_max - X_min'''
def plot_Normalized_Responsivity_curve(Responsivity,volts_list,wavelength,sample_number):
    Normalized_Responsivity = (Responsivity - Responsivity.min()) / \
                              (Responsivity.max() - Responsivity.min())
    #print(Normalized_Responsivity)

    figure, ax = plt.subplots(figsize=(12, 9))  # 创建一个画板
    X_Values = wavelength.values
    # Y_Values = []
    X_range = np.arange(min(wavelength.values), max(wavelength.values), 50)
    #print(X_range)
    for i in volts_list:
        #     Y_Values.append(Responsivity[i])i
        #     Y_Values.append(Responsivity[i])
        plt.plot(X_Values, abs(Normalized_Responsivity[i]),
                 label=(format(str(i) + "V")), linewidth=10)
    # ax.cla()
    # plt.pause(0.1)

    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": 12,  # 设置字体大小
        "font.weight": 'bold',  # 设置字体加粗

        "mathtext.fontset": 'Times New Roman',
    }
    title_text = sample_number
    # plt.scatter(X_Values, Responsivity[], color='black', s=100, cmap='viridis', label='dark', linewidths=3)
    # plt.scatter(X_Values, Responsivity[1], color='b', s=100, cmap='viridis', marker='*', label=pc_wavelength + 'nm',linewidths=3)
    # 设置标题
    plt.title(title_text, fontdict={'family': 'Times New Roman', 'size': 30, 'weight': 'bold'})  # 设置标题
    # 设置刻度
    plt.yticks(fontproperties='Times New Roman', weight='bold', size=30)
    plt.xticks(X_range, fontproperties='Times New Roman', weight='bold', size=30)  # 设置横轴图像刻度
    # 设置横纵轴说明
    plt.xlabel("Wavelength/(λ)", fontproperties='Times New Roman', weight='bold', size=30)  # 设置X轴图示，字体，加粗
    plt.ylabel("Responsivity/(A/W)", fontproperties='Times New Roman', weight='bold', size=30)
    # 设置图例

    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 20,
             }
    plt.legend(loc='upper right', frameon=False, fontsize=20, prop=font1)  # 设置图例在右下角显示，取消边框
    # 设置线宽
    ax_width = 3.5
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    # 设置刻度朝向
    ax.tick_params(axis='both', which='both', direction='in', size=5, colors='black', width=3,
                   pad=15)  # pad是调整刻度线与刻度之间的距离

    # ax.set_yscale("log")

    # plt.style.use('ggplot')
    # print("*"*10+'这是电压值'+"*"*10+'\n',"\n".join(str(x) for x in X_V.values))
    # print("*"*10+'这是暗电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_dark_I.values))
    # print("*"*10+'这是光电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_uv_I.values))
    plt.show()

    return None

'''绘制紫外/可见光抑制比曲线
参数：
    紫外/可见光抑制比DataFrame -> Rejection_ratio'''
def draw_Rejection_ratio_curve(Rejection_ratio):
    figure, ax = plt.subplots(figsize=(12, 9))  # 创建一个画板
    X_Values = Rejection_ratio.index
    Y_Values = Rejection_ratio.values
    print(X_Values,Y_Values)
    X_range = np.arange(min(Rejection_ratio.index), max(Rejection_ratio.index)+1, 1)

    title_text = sample_number
    plt.scatter(X_Values, Y_Values, color='black', s=100, cmap='viridis', label='Rejection ratio', linewidths=3)
    # plt.scatter(X_Values, Responsivity[1], color='b', s=100, cmap='viridis', marker='*', label=pc_wavelength + 'nm',linewidths=3)
    # 设置标题
    plt.title(title_text, fontdict={'family': 'Times New Roman', 'size': 30, 'weight': 'bold'})  # 设置标题
    # 设置刻度
    plt.yticks(fontproperties='Times New Roman', weight='bold', size=30)
    plt.xticks(X_range, fontproperties='Times New Roman', weight='bold', size=30)  # 设置横轴图像刻度
    # 设置横纵轴说明
    plt.xlabel("Bias/(V)", fontproperties='Times New Roman', weight='bold', size=30)  # 设置X轴图示，字体，加粗
    plt.ylabel("Rejection ratio/a.u", fontproperties='Times New Roman', weight='bold', size=30)
    # 设置图例

    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 25,
             }
    plt.legend(loc='upper right', frameon=False, fontsize=30, prop=font1)  # 设置图例在右下角显示，取消边框
    # 设置线宽
    ax_width = 3.5
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    # 设置刻度朝向
    ax.tick_params(axis='both', which='both', direction='in', size=5, colors='black', width=3,
                   pad=15)  # pad是调整刻度线与刻度之间的距离

    # ax.set_yscale("log")

    # plt.style.use('ggplot')
    # print("*"*10+'这是电压值'+"*"*10+'\n',"\n".join(str(x) for x in X_V.values))
    # print("*"*10+'这是暗电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_dark_I.values))
    # print("*"*10+'这是光电流值'+"*"*10+'\n',"\n".join(str(x) for x in Y_uv_I.values))
    plt.show()


    return None

'''绘制PDCR - light_dendity -volts曲线
参数：
    特点电压点列随光功率密度变化曲线 -> PDCR_df_at_Volts_and_pc_density
    不含波长的光功率密度 -> pc_density_without_wavelength
返回值：None
'''
def draw_PDCR_light_dendity_volts_curve(PDCR_df_at_Volts_and_pc_density,pc_density_without_wavelength,volts_list):
    figure, ax = plt.subplots(figsize=(12, 9))  # 创建一个画板
    X_Values = pc_density_without_wavelength
    # Y_Values = []
    #X_range = np.arange(min(pc_density_without_wavelength), max(pc_density_without_wavelength))
    #print(X_range)
    for i in volts_list:
        plt.plot(X_Values, PDCR_df_at_Volts_and_pc_density[i],
                 label=(format(str(i) + "V")), linewidth=10)
    # ax.cla()
    # plt.pause(0.1)

    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": 12,  # 设置字体大小
        "font.weight": 'bold',  # 设置字体加粗

        "mathtext.fontset": 'Times New Roman',
    }
    title_text = sample_number
    # plt.scatter(X_Values, Responsivity[], color='black', s=100, cmap='viridis', label='dark', linewidths=3)
    # plt.scatter(X_Values, Responsivity[1], color='b', s=100, cmap='viridis', marker='*', label=pc_wavelength + 'nm',linewidths=3)
    # 设置标题
    plt.title(title_text, fontdict={'family': 'Times New Roman', 'size': 30, 'weight': 'bold'})  # 设置标题
    # 设置刻度
    plt.yticks(fontproperties='Times New Roman', weight='bold', size=30)
    plt.xticks(fontproperties='Times New Roman', weight='bold', size=30)  # 设置横轴图像刻度
    # 设置横纵轴说明
    plt.xlabel("Light intensity/(W/mm2)", fontproperties='Times New Roman', weight='bold', size=30)  # 设置X轴图示，字体，加粗
    plt.ylabel("PDCR", fontproperties='Times New Roman', weight='bold', size=30)
    # 设置图例

    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 20,
             }
    plt.legend(loc='upper right', frameon=False, fontsize=20, prop=font1)  # 设置图例在右下角显示，取消边框
    # 设置线宽
    ax_width = 3.5
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    # 设置刻度朝向
    ax.tick_params(axis='both', which='both', direction='in', size=5, colors='black', width=3,
                   pad=15)  # pad是调整刻度线与刻度之间的距离

    # ax.set_yscale("log")

    plt.style.use('ggplot')

    plt.show()

    return

'''绘制探测度曲线
参数：
    探测度数据 = '''
def draw_Detectivity_curve(Detectivity):
    figure, ax = plt.subplots(figsize=(12, 9))  # 创建一个画板
    X_Values = wavelength.values
    # Y_Values = []
    X_range = np.arange(min(wavelength.values), max(wavelength.values), 50)
    # print(X_range)
    for i in volts_list:
        #     Y_Values.append(Responsivity[i])i
        #     Y_Values.append(Responsivity[i])
        plt.plot(X_Values, abs(Detectivity[i]),
                 label=(format(str(i) + "V")), linewidth=10)
    # ax.cla()
    # plt.pause(0.1)

    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": 12,  # 设置字体大小
        "font.weight": 'bold',  # 设置字体加粗

        "mathtext.fontset": 'Times New Roman',
    }
    title_text = sample_number
    # plt.scatter(X_Values, Responsivity[], color='black', s=100, cmap='viridis', label='dark', linewidths=3)
    # plt.scatter(X_Values, Responsivity[1], color='b', s=100, cmap='viridis', marker='*', label=pc_wavelength + 'nm',linewidths=3)
    # 设置标题
    plt.title(title_text, fontdict={'family': 'Times New Roman', 'size': 30, 'weight': 'bold'})  # 设置标题
    # 设置刻度
    plt.yticks(fontproperties='Times New Roman', weight='bold', size=30)
    plt.xticks(X_range, fontproperties='Times New Roman', weight='bold', size=30)  # 设置横轴图像刻度
    # 设置横纵轴说明
    plt.xlabel("Wavelength/(λ)", fontproperties='Times New Roman', weight='bold', size=30)  # 设置X轴图示，字体，加粗
    plt.ylabel("Detetivity/(Jones)", fontproperties='Times New Roman', weight='bold', size=30)
    # 设置图例

    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 20,
             }
    plt.legend(loc='upper right', frameon=False, fontsize=20, prop=font1)  # 设置图例在右下角显示，取消边框
    # 设置线宽
    ax_width = 3.5
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    # 设置刻度朝向
    ax.tick_params(axis='both', which='both', direction='in', size=5, colors='black', width=3,
                   pad=15)  # pad是调整刻度线与刻度之间的距离

    ax.set_yscale("log")

    plt.show()

    return

if __name__ == '__main__':

    '''传入文件夹名称'''
    dir_name = 'experiment4\\02'

    '''获得一次实验所有样品编号'''

    '''传入光功率密度文件名'''
    pc_density_table_name = 'pc_density_table.csv'

    '''遍历文件夹获取不同波长下光电流数据，以及暗电流数据'''
    pc_data_array,dark_data_array,sample_number = data_to_IV(dir_name)

    '''获取紫外辐射下探测器电压电流数据'''
    pc_current_data, pc_wavelength = get_pc_current(dir_name, '210')

    '''绘制暗电流与光电流I-V曲线'''
    #volt_dark_pc = draw_light_dark_IV_curve(dark_data_array, pc_current_data, pc_wavelength, sample_number)

    '''绘制半对数坐标系下暗电流与光电流I-V曲线'''
    #semilog_volt_dark_pc = draw_semilog_light_dark_curve(dark_data_array, pc_current_data, pc_wavelength, sample_number)

    '''获得探测器测试电压'''
    volts_as_index = get_volts_as_index_array(dark_data_array)

    '''获得光功率密度'''
    pc_density,pc_density_without_wavelength,wavelength = get_pc_density(pc_density_table_name)

    '''传入探测器有效光敏面积'''
    effective_area = 0.2635

    '''计算探测器上光功率（W）： 探测器上光功率 = 光功率密度 * 有效光敏面积'''
    detector_acquired_optical_power= get_detector_acquired_optical_power(effective_area,pc_density_without_wavelength)

    '''计算净光电流'''
    net_pc_data_array = get_net_pc_data(pc_data_array,dark_data_array)

    '''电压间隔step'''
    volt_step = 0.1

    '''需要的特定电压下的电压值'''
    volts_list = [0,1,2,3,4,5]

    '''获得特定电压下的净光电流列索引'''
    net_pc_columns_index = get_net_pc_columns_index(volts_list,volt_step)

    '''获得特定电压下净光电流'''
    net_pc_at_volts = get_net_pc_at_volts(net_pc_data_array,net_pc_columns_index,volts_as_index)

    '''计算PDCR'''
    PDCR_df_at_Volts_and_pc_density = calculate_PDCR_df_at_Volts_and_pc_density(pc_data_array, dark_data_array,volts_list,pc_density_without_wavelength)

    '''计算响应度'''
    Responsivity = calculate_Responsivity(net_pc_at_volts,detector_acquired_optical_power,wavelength,volts_list)

    #Responsivityat_0V_of_all_sample = calculate_Responsivityat_0V_of_all_sample(dir_name)

    '''计算紫外/可见光抑制比时传入的紫外线波长'''
    UV_wavelength = 210

    '''计算紫外/可见抑制比'''
    Rejection_ratio = calculate_Rejection_ratio(Responsivity,UV_wavelength = '210')

    '''计算等效噪声功率'''
    Detectivity = cauculate_Detectivity(dark_data_array, Responsivity,volts_list,UV_wavelength,effective_area)

    '''绘制响应度曲线'''
    #plot_Responsivity_curve(Responsivity,volts_list,wavelength,sample_number)

    '''绘制半对数坐标响应度曲线'''
    #plot_semilog_Responsivity_curve(Responsivity, volts_list, wavelength, sample_number)

    '''绘制归一化响应度曲线'''
    #plot_Normalized_Responsivity_curve(Responsivity, volts_list, wavelength, sample_number)

    '''绘制紫外/可见光抑制比曲线'''
    #draw_Rejection_ratio_curve(Rejection_ratio)

    '''绘制PDCR - light_dendity -volts曲线'''
    #draw_PDCR_light_dendity_volts_curve(PDCR_df_at_Volts_and_pc_density,pc_density_without_wavelength,volts_list)

    '''绘制探测度曲线随波长变化曲线'''
    #draw_Detectivity_curve(Detectivity)