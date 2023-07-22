import xlrd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scipy
import seaborn as sns
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# def read_f15():
path= 'f15.xls'
data = xlrd.open_workbook(path)
table = data.sheets()[0]
nrows = table.nrows  # 行数
ncols = table.ncols  # 列数
datamatrix = np.zeros((nrows-1, 1))
title = table.row_values(0,0,5)
note=[]
for i in range(nrows-1):
    rows = table.row_values(i+1,0,5)
    rows = list(map(lambda x: np.nan if x == '' else x, rows))#缺失值处理
    datamatrix[i, :] = rows[-1:]
    note.append(rows[:4])
# plt.figure("Box_figure")
# plt.title('Box_figure')
# plt.boxplot(datamatrix[:,0], labels=[1])  # 绘制箱线图
note=np.array(note)
intensity=['NG','LGI','MGI','HGI']
for i in range(4):
    plt.figure(intensity[i])
    plt.title(intensity[i])
    plt.xlabel('time(month)')
    plt.ylabel('plants(g)')
    for j in range(3):
        idx=i*3+j
        plt.plot(datamatrix[list(range(idx,300,12)),0],label=note[idx,3])
        # plt.boxplot(datamatrix[list(range(idx, 300, 12)), 0])
    plt.legend()

new_data=np.zeros((12,5))
for i in range(12):
    temp_block=note[i,3]
    for j in range(5):
        temp_year=note[60*j,0]
        new_data[i,j]=datamatrix[np.where((note[:,3]==temp_block) & (note[:,0]==temp_year)),0].mean()


plt.figure('plants-graz_inten')
plt.title('plants-graz_inten')
plt.xlabel('graz_inten(x羊/天/公顷)')
plt.ylabel('delta_plants(g)')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
delta_new_data = new_data[:, 1:] - new_data[:, 0:-1]
x=[0,2,4,8]
y=[delta_new_data[0:3,:].mean(),delta_new_data[3:6,:].mean(),delta_new_data[6:9,:].mean(),delta_new_data[9:12,:].mean()]
coef_plants = np.polyfit(x, y, 2)
plt.scatter(x,y)
temp_plants=[]
for i in np.arange(0,10,0.2):
    temp_plants.append(np.polyval(coef_plants, i))
plt.plot(np.arange(0,10,0.2),temp_plants)
np.save('coef_plants.npy',coef_plants)



    # return title,note,datamatrix