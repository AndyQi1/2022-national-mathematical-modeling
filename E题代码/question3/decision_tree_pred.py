import xlrd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import scipy
import seaborn as sns
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor



def read_f14():
    path= 'f14.xlsx'
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows-1, 5))
    title = table.row_values(0,0,8)
    note=[]
    for i in range(nrows-1):
        rows = table.row_values(i+1,0,8)
        rows = list(map(lambda x: np.nan if x == '' else x, rows))#缺失值处理
        datamatrix[i, :] = rows[-5:]
        note.append(rows[:3])
    plt.figure("Box_figure")
    plt.title('Box_figure')
    plt.boxplot(datamatrix[:, [0,1,3]], labels=['SOC','SIC','FN'])  # 绘制箱线图

    table = data.sheets()[1]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows - 1, ncols-3))
    title = table.row_values(0, 0, ncols)
    note = []
    for i in range(nrows - 1):
        rows = table.row_values(i + 1, 0, ncols)
        rows = list(map(lambda x: np.nan if x == '' else x, rows))  # 缺失值处理
        datamatrix[i, :] = rows[-(ncols-3):]
        note.append(rows[:3])
    note=np.array(note)
    return title,note,datamatrix

def plot_f14():
    for i in range(4):
        plt.figure(note[i * 3][2])
        plt.title(note[i * 3][2])
        for j in range(3):
            plt.subplot(1, 3, j + 1)
            plt.title(note[3 * i + j][1] + "({})".format(note[i * 3][2]))
            x = []
            y = []
            for k in range(5):
                x.append(int(float(note[3 * i + j + k * 12][0])))
                y.append(data[3 * i + j + k * 12, :])
            y = np.array(y)
            x = np.array(x)
            plt.plot(x, y[:, 0], label='SOC')
            plt.plot(x, y[:, 1], label='SIC')
            plt.plot(x, y[:, 2], label='STC')
            plt.plot(x, y[:, 3], label='TN')
            plt.plot(x, y[:, 4], label='C/N')
            plt.legend()

title,note,data=read_f14()
plot_f14()

pred_2022_x=np.load('pred_2022_x.npy')

pred_data=np.zeros((12,5))
i=0
for block in note[0:12,1]:
    fore_data=data[np.where(note[:,1]==block)[0]]
    x=fore_data[:,[5,6,7]]
    y=fore_data[:,0]
    model=DecisionTreeRegressor(random_state=0)
    model.fit(x,y)
    pred_data[i,0]=model.predict(pred_2022_x[1:].reshape(1,-1))

    y = fore_data[:, 1]
    model = DecisionTreeRegressor(random_state=0)
    model.fit(x, y)
    pred_data[i, 1] = model.predict(pred_2022_x[1:].reshape(1, -1))

    y = fore_data[:, 3]
    model = DecisionTreeRegressor(random_state=0)
    model.fit(x, y)
    pred_data[i, 3] = model.predict(pred_2022_x[1:].reshape(1, -1))

    pred_data[i, 2]=pred_data[i, 0]+pred_data[i, 1]
    pred_data[i, 4]=pred_data[i,2]/pred_data[i,3]
    i=i+1
np.save('pred_data.npy',pred_data)