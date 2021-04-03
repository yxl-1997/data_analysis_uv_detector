# -*- coding: utf-8 -*-
# @Time : 2021/3/29 20:36
# @Author : yxl
# @File : deno3.py
# @Project : data_analysis_uv_detector
# -*- coding: utf-8 -*-
# @Time : 2021/1/5 15:21
# @Author : yxl
# @File : R_0_to_1Volts.py
# @Project : slef-power-uv-detector

import re
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  #字体管理器
import xlsxwriter
import math



#打印数组的全部数据
np.set_printoptions(threshold=np.inf)


#暗电流/210nm光电流
pc_210nm_path=r"C:\Users\35119\Desktop\experiment4\02\210.csv"

dark_current_path=r"C:\Users\35119\Desktop\experiment4\02\dark.csv"

#读取2号样品csv 文件
pc_path = r'C:\Users\35119\Desktop\experiment2\01'
files = os.listdir(pc_path)
files_csv = list(filter(lambda x: x[-4:]=='.csv' , files))
print(len(files_csv))


#暗电流数据获取并装入数组
dark_path = r'C:\Users\35119\Desktop\experiment2\01\dark.csv'
dark_data = pd.read_csv(dark_path,skiprows=224,usecols=(1,2),error_bad_lines=False)
dark_array = np.array(dark_data)
#print('***********以下是暗电流矩阵************')
#print(dark_array,dark_array.shape)


#光电流表波长及光功率密度提取并装入数组
w_path=r'C:\Users\35119\Desktop\uv_detector\Optical_power_density.csv'
w=pd.read_csv(w_path,skiprows=0,usecols=(0,4),encoding='gbk')
w_array=np.array(w)
#print(w_array)


#读取不同波长w_l(wavelength)
wavelength=w_array[:,:1]
#print(wavelength)
#print(wavelength.shape)

print('#######'*5)

# # 循环读取文件中‘V','I'两列，并装入矩阵
pc_data_list = []
for file in  files_csv:
    file_name = pc_path+"/"+file
    tmp = np.loadtxt(file_name,delimiter=',',skiprows=225,usecols=(1,2),encoding='utf-8')
    pc_data_list.append(tmp)
pc_array = np.array(pc_data_list)
#print('*8' * 10 + '以下是光电流矩阵' + '*8' *10)
#print(pc_array,pc_array.shape)


pc_array_v_integer = pc_array[:37,:101:10,:]         #提取每组光电流下整数[-5,-4,-3,-2,-1,0,1,2,3,4,5]V光电流数据
#print(pc_array_v_integer)
#print(pc_array_v_integer.shape)


#合并所需电压下光电流数据[0,1,2,3,4,5]V光电流数据
pc_array_v_integer_positive = pc_array_v_integer[:,5:,:]    #提取正整数光电流数据
#print(pc_array_v_integer_positive)
#print(pc_array_v_integer_positive.shape)


#提取暗电流[0,1,2,3,4,5]V数据
dark_array_integer = dark_array[:101:10,:2]                 #提取整数暗电流数据
#print(dark_array_integer)
dark_array_integer_0 = dark_array_integer[5:6,:]            #提取0V暗电流
dark_array_integer_1 = dark_array_integer[6:7,:]            #提取1V暗电流
dark_array_integer_2 = dark_array_integer[7:8,:]            #提取2V暗电流
dark_array_integer_3 = dark_array_integer[8:9,:]            #提取3V暗电流
dark_array_integer_4 = dark_array_integer[9:10,:]            #提取4V暗电流
dark_array_integer_5 = dark_array_integer[10:11,:]            #提取5V暗电流
#print('*' * 10 + '以下是dark_array_integer--整数电压下暗电流数值' + '*' *10)
#print(dark_array_integer,dark_array_integer.shape)
#print('*' * 10 + '以上是0V下dark_array_integer' + '*' *10)
#print(dark_array_integer_0[:,1:2])






#合并所需电压下光电流数据
#print(pc_array_v_integer_positive)
pc_array_v_0 = np.concatenate(pc_array_v_integer_positive[:,:1,:])      #提取0V光电流数据
#print(pc_array_v_0)
pc_array_v_1 = np.concatenate(pc_array_v_integer_positive[:,1:2,:])     #提取1V光电流数据
#print(pc_array_v_1)
pc_array_v_2 = np.concatenate(pc_array_v_integer_positive[:,2:3,:])     #提取2V光电流数据
pc_array_v_3 = np.concatenate(pc_array_v_integer_positive[:,3:4,:])     #提取3V光电流数据
pc_array_v_4 = np.concatenate(pc_array_v_integer_positive[:,4:5,:])     #提取4V光电流数据
pc_array_v_5 = np.concatenate(pc_array_v_integer_positive[:,5:6,:])     #提取5V光电流数据
#print(pc_array_v_2)
#print('*' * 10 + '以下是光电流矩阵' + '*' *10)
#print(pc_array_v_0,pc_array_v_0.shape)
#print(pc_array_v_0[:,1:])



#计算净光电流Net_photo_current（净光电流）矩阵
net_photo_current_0 = pc_array_v_0 - dark_array_integer_0               #光电流数值-暗电流数值
net_photo_current_1 = pc_array_v_1 - dark_array_integer_1               #光电流数值-暗电流数值
net_photo_current_2 = pc_array_v_2 - dark_array_integer_2
net_photo_current_3 = pc_array_v_3 - dark_array_integer_3
net_photo_current_4 = pc_array_v_4 - dark_array_integer_4
net_photo_current_5 = pc_array_v_5 - dark_array_integer_5
#print('*' * 10 + '以下是净光电流' + '*' *10)
#print(net_photo_current_0,net_photo_current_0.shape)
#print(pc_array_v_0,dark_array_integer_0,dark_array_integer_0.shape)

#计算探测器上光功率
A = 0.2635
wA=(w_array[:, 1:]*A)#光功率密度*有效光敏面积=探测器上光功率
#print(wA)
#print(wA.shape)
#计算响应度

R_0 = net_photo_current_0[:,1:]/wA      #只保留响应度，去掉波长
R_1 = net_photo_current_1[:,1:]/wA
R_2 = net_photo_current_2[:,1:]/wA
R_3 = net_photo_current_3[:,1:]/wA
R_4 = net_photo_current_4[:,1:]/wA
R_5 = net_photo_current_5[:,1:]/wA
#for index, i in np.ndenumerate(R_0):#打印index
print('*' * 10 + '0V下探测器响应度' + '*' *10)

for i in np.array(abs(R_0)):
    R_0_normalized_array = []
    print("%e"%i)#以科学计数法打印
#R_0_data = pd.DataFrame(R_0)

'''归一化响应度：
x_normalization =  (x - Min)/(Max - Min)'''
print('*'*20+ '以下是0v归一化响应度'+ '*'*20)
for value in R_0:
    R_0_normalized = (value - R_0.min()) / (R_0.max() - R_0.min())
    R_0_normalized_array.append(R_0_normalized)
    print('%e'%R_0_normalized)


plt.plot(wavelength,R_0_normalized_array)
plt.show()


#绘制响应度曲线
fig = plt.figure(dpi=200)
figure, ax = plt.subplots(figsize=(12,9))#dpi (Dot per inch)
#plt.plot(wavelength,R_0,color="red", linewidth=3.5, linestyle="-", label="0V",marker='_',markersize=10)
plt.plot(wavelength,np.abs(R_0),color="red", linewidth=3.5, linestyle="-", label="0V",marker='s',markersize=10)#fig = plt.figure(num=1, figsize=(16,9),dpi=200)
plt.plot(wavelength,np.abs(R_1),color="blue", linewidth=3.5, linestyle="-", label="1V",marker='^',markersize=10)
plt.plot(wavelength,np.abs(R_2),color="green", linewidth=3.5, linestyle="-", label="2V",marker='v',markersize=10)
plt.plot(wavelength,np.abs(R_3),color="black", linewidth=3.5, linestyle="-", label="3V",marker='<',markersize=10)
plt.plot(wavelength,np.abs(R_4),color="orange", linewidth=3.5, linestyle="-", label="4V",marker='>',markersize=10)
plt.plot(wavelength,np.abs(R_5),color="violet", linewidth=3.5, linestyle="-", label="5V",marker='o',markersize=10)
plt.ylabel('Responsivity(A/W)', fontdict={'family' : 'Times New Roman', 'size'   : 23,'weight':'heavy'})
plt.xlabel('Wavelength(nm)', fontdict={'family' : 'Times New Roman', 'size'   : 23,'weight':'heavy'})
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 23})
#plt.title('sample 1',fontdict={'family' : 'Times New Roman', 'size':23, 'weight':'normal'})
plt.yticks(fontproperties = 'Times New Roman', weight='normal',size = 23)
plt.xticks(fontproperties = 'Times New Roman', weight='normal',size = 23)
ax_width=3.5
ax.spines['bottom'].set_linewidth(ax_width)
ax.spines['left'].set_linewidth(ax_width)
ax.spines['top'].set_linewidth(ax_width)
ax.spines['right'].set_linewidth(ax_width)
plt.show()
#print(R_0.shape,R_1.shape,R_2.shape,R_3.shape,R_4.shape,R_5.shape)


#计算UV\vis抑制比=R_210/R_400
#Rejection_ratio = R_210/R_400
#0伏抑制比
Rejection_ratio_R_0 =np.abs((R_0[2])/(R_0[23]))
Rejection_ratio_R_1 = (R_1[2])/(R_1[23])
Rejection_ratio_R_2 = (R_2[2])/(R_2[23])
Rejection_ratio_R_3 = (R_3[2])/(R_3[23])
Rejection_ratio_R_4 = (R_4[2])/(R_4[23])
Rejection_ratio_R_5 = (R_5[2])/(R_5[23])
print('*' * 10 + '以下是0-1V_UV/VIS抑制比' + '*' *10)
print('0V:',"%e"%Rejection_ratio_R_0,'1V:',"%e"%Rejection_ratio_R_1,
      '2V:',"%e"%Rejection_ratio_R_2,'3V:',"%e"%Rejection_ratio_R_3,
      '4V:',"%e"%Rejection_ratio_R_4,'5V:',"%e"%Rejection_ratio_R_5)

'''Detectivity (D*) is one of the key figure-of-merits for a photode- tector, 
which is used to describe the sensitivity.
 D* can be calculated using the following equation
  D= R*     D越大,探测器的性能越好'''
#q is the electronic charge 1.6 *pow(10,19)
#Jd is the dark current intensity
#
q = math.pow(10,-19)*1.6
J_0 = dark_array_integer_0[:,1:2]/A
J_1 = dark_array_integer_1[:,1:2]/A
J_2 = dark_array_integer_2[:,1:2]/A
J_3 = dark_array_integer_3[:,1:2]/A
J_4 = dark_array_integer_4[:,1:2]/A
J_5 = dark_array_integer_5[:,1:2]/A

D_0= R_0[2]/math.sqrt(2*q*J_0)
D_1= R_1[2]/math.sqrt(2*q*J_1)
D_2= R_2[2]/math.sqrt(2*q*J_2)
D_3= R_3[2]/math.sqrt(2*q*J_3)
D_4= R_4[2]/math.sqrt(2*q*J_4)
D_5= R_5[2]/math.sqrt(2*q*J_5)
print('*'*10 + '以下是0-1V探测器的探测度-Jones 探测度越大，探测器的性能越好' +'*'*10)
print('0V:',"%e"%D_0,'1V:',"%e"%D_1,'2V:',"%e"%D_2,
      '3V:',"%e"%D_3,'4V:',"%e"%D_4,'5V:',"%e"%D_5,'单位：Jones')


'''linear dynamic rang (LDR)线性动态范围计算公式:
 LDR =20log(Iph-Id) 
计算0V下LDR'''

#print(pc_array[2:3,:,1:])#210nm光电流数值
#print(dark_array[:,1:])#暗电流数值
#print(pc_array_v_0,pc_array_v_0.shape)
Iph_0 = pc_array_v_0[2:3,1:]
Id_0 =  dark_array_integer_0[:,1:]
LDR_0= np.array(math.log( abs(Iph_0/Id_0)) * 20)
    #print(LDR_array)
print('*' * 10 + '以下是0V-210nm下探测器的线性动态范围' + '*' * 10)
print(LDR_0, 'dB')


'''(External Quantum Efficiency)EQE = (h*c*R)/qλ*100%
 where c denotes the light velocity, h denotes Planck’s constant,
  and λ denotes the light wavelength'''
print('*' * 10 + '以下是0-5V探测器外量子效率' + '*' * 10)
h = math.pow(10,-34)*6.626 #单位:J·s
c = math.pow(10,8)*3

EQE = h*c*R_0  / (q*math.pow(10,-9)*wavelength)
print(EQE)



#从桌面读取实验数据
#pc_210nm_path=r"C:\Users\35119\Desktop\2020-11-24\left_up_1\220.csv"

#dark_current_path=r"C:\Users\35119\Desktop\2020-11-24\left_up_1\dark.csv"
data210=np.loadtxt(pc_210nm_path,delimiter=",",skiprows=225,usecols=(1,2),encoding='utf-8')
datadark02=np.loadtxt(dark_current_path,delimiter=",",skiprows=225,usecols=(1,2),encoding='utf-8')
pc_02_210nm=np.loadtxt(pc_210nm_path,delimiter=",",skiprows=225,usecols=(1,2),encoding='utf-8')


#绘图
x2=np.array(datadark02[:,:-1])
#print('**********以下是电压值************')
#for i in x2:
    #print(i)
    #print('%.2f'%i)

ydark02=np.array((datadark02[:,1:]))/(pow(10,-9))
y210_02=np.array((pc_02_210nm[:,1:]))/(pow(10,-9))
#对数组取绝对值
y_dark_abs = np.array(abs((datadark02[:,1:])))/(pow(10,-9))
print('***********以下是暗电流对数值***********')
for a in y_dark_abs:
    print('%e'%a)

y210_02_abs = np.array(abs((pc_02_210nm[:,1:])))/(pow(10,-9))
print('************以下是光电流对数值******************')
for b in y210_02_abs:
    print('%e'%b)

#y_log_dark = np.array([math.log(a,10)for a in y_dark_abs])
#y_log_210_2 = np.array([math.log(b,10)for b in y210_02_abs])
# 设置输出的图片大小
fig = plt.figure(num=1,dpi=200)
figure, ax = plt.subplots(figsize=(12,9))#dpi (Dot per inch)

# 在同一幅图片上画两条折线
#A, = plt.plot(x2, ydark02, '-', color='000000'   ,label='dark', linewidth=5.0,marker='s',markersize=10)
#B, = plt.plot(x2, y210_02, '-',color='blue',label='210nm', linewidth=5.0,marker='>',markersize=10)
C, = plt.semilogy(x2,y_dark_abs,'-', color='000000'   ,label='dark', linewidth=5.0,marker='s',markersize=10)
D, = plt.semilogy(x2,y210_02_abs,'-', color='blue'   ,label='210', linewidth=5.0,marker='>',markersize=10)


# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman','we ight': 'black','size': 30}
#legend = plt.legend(handles=[C,D], prop=font1,frameon=False)#去掉图例边框)

# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)
labels = ax.get_xticklabels( ) + ax.get_yticklabels( )
[label.set_fontname('Times New Roman') for label in labels]
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 23})
# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman','weight': 'black','size': 30}
plt.xlabel('Volt(V)', font2)
plt.ylabel('Current(nA)', font2)
ax_width=3.5
ax.spines['bottom'].set_linewidth(ax_width)
ax.spines['left'].set_linewidth(ax_width)
ax.spines['top'].set_linewidth(ax_width)
ax.spines['right'].set_linewidth(ax_width)
#plt.grid( color = 'black',linestyle='-.',linewidth = 2)
# 将文件保存至文件中并且画出图
#plt.savefig('figure.eps')
#plt.semilogy(x2,y_dark_abs)
#plt.semilogy(x2,y210_02_abs)
plt.show()