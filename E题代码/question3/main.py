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
coef_lst=[['pred_value','intercept','tempe_coef','humi10_coef','humi40_coef']]
all_x=[]
all_soc_y=[]
all_soc_dy_y=[]
all_sic_y=[]
all_sic_dy_y=[]
all_tn_y=[]
all_tn_dy_y=[]
#多元线性回归
for block in note[0:12,1]:
    fore_data=data[np.where(note[:,1]==block)[0]]
    fore_data_x=data[np.where(note[:,1]==block)[0]][:,[8,5,6,7]]
    fore_data_x=fore_data_x[1:,:]
    all_x.append(fore_data_x.copy())
    fore_data_x=fore_data_x[:,1:]
    fore_data_x=np.vstack((fore_data_x,fore_data_x))
    col = ['tempe','humi10','humi40']
    data_pd = pd.DataFrame(fore_data_x,columns=col)
    y_label_lst=['d_Soc','d_SIC','d_TN']
    y_idx=[0,1,3]
    for i in range(3):
        idx=y_idx[i]
        y_label=y_label_lst[i]
        f1=fore_data[1:,idx]
        f2=fore_data[0:-1,idx]
        df=f1-f2
        if i==0:
            all_soc_y.append(df)
            all_soc_dy_y.append(df*f2)
        elif i==1:
            all_sic_y.append(df)
            all_sic_dy_y.append(df*f2)
        else:
            all_tn_y.append(df)
            all_tn_dy_y.append(df*f2)
        data_pd[y_label]=np.hstack((df,df))
        mod = smf.ols(formula='{}~tempe+humi10+humi40'.format(y_label),data=data_pd)
        res = mod.fit()
        # print(res.summary())
        # print("rmse:", sqrt(mean_squared_error(res.predict(), data_pd[y_label])))
        print(res.params.values)
        coef_lst.append([block+'-'+y_label,res.params.values])

#不同放牧策略
y_label_lst=['d_Soc_all','d_SIC_all','d_TN_all']
all_x=np.vstack(all_x)
all_soc_y=np.hstack(all_soc_y)
all_sic_y=np.hstack(all_sic_y)
all_tn_y=np.hstack(all_tn_y)
all_soc_dy_y=np.hstack(all_soc_dy_y)
all_sic_dy_y=np.hstack(all_sic_dy_y)
all_tn_dy_y=np.hstack(all_tn_dy_y)
# all_y=[all_soc_y,all_sic_y,all_tn_y]
# col = ['graz_itst','tempe', 'humi10', 'humi40']
# data_pd = pd.DataFrame(all_x, columns=col)
# all_coef_lst=[['pred_value','intercept','graz_itst_coef','tempe_coef','humi10_coef','humi40_coef']]
# print('~~~~~~~~~~all_pred~~~~~~~~~~~\n')
# for i in range(3):
#     y_label=y_label_lst[i]
#     data_pd[y_label]=all_y[i]
#     mod = smf.ols(formula='{}~graz_itst+tempe+humi10+humi40'.format(y_label), data=data_pd)
#     res = mod.fit()
#     print(res.params.values)
#     print("rmse:", sqrt(mean_squared_error(res.predict(), data_pd[y_label])))
#     all_coef_lst.append([y_label, res.params.values])

pred_2022_x=np.load('pred_2022_x.npy')
# #predict
# pred_data=np.zeros((12,5))
# for i in range(12):
#     t_lst=[0,1,3]
#     for j in range(3):
#         pred_2022_y=(coef_lst[i*3+j+1][1]*pred_2022_x).sum()+data[48+i,t_lst[j]]
#         pred_data[i,t_lst[j]]=pred_2022_y
#     pred_data[i,2]=pred_data[i,0]+pred_data[i,1]
#     pred_data[i, 4] = pred_data[i, 2] / pred_data[i, 3]
# np.save('pred_data.npy',pred_data)

plt.figure('soc-graz_inten')
plt.title('soc-graz_inten')
plt.xlabel('graz_inten(x羊/天/公顷)')
plt.ylabel('delta_soc*soc')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
soc_x=np.array([0,2,4,8])
soc_y=np.zeros(4)
soc_y[0]=all_soc_dy_y[0:12].sum()/12.0
soc_y[1]=all_soc_dy_y[12:24].sum()/12.0
soc_y[2]=all_soc_dy_y[24:36].sum()/12.0
soc_y[3]=all_soc_dy_y[36:48].sum()/12.0
plt.scatter(soc_x,soc_y)
coef_soc = np.polyfit(soc_x, soc_y, 3)
temp_soc=[]
for i in np.arange(0,10,0.2):
    temp_soc.append(np.polyval(coef_soc, i))
plt.plot(np.arange(0,10,0.2),temp_soc)

plt.figure('tn-graz_inten')
plt.title('tn-graz_inten')
plt.xlabel('graz_inten(x羊/天/公顷)')
plt.ylabel('delta_tn*tn')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
tn_y=np.zeros(4)
tn_y[0]=all_tn_dy_y[0:12].sum()/12.0
tn_y[1]=all_tn_dy_y[12:24].sum()/12.0
tn_y[2]=all_tn_dy_y[24:36].sum()/12.0
tn_y[3]=all_tn_dy_y[36:48].sum()/12.0
plt.scatter(soc_x,tn_y)
coef_tn = np.polyfit(soc_x, tn_y, 3)
temp_tn=[]
for i in np.arange(0,10,0.2):
    temp_tn.append(np.polyval(coef_tn, i))
plt.plot(np.arange(0,10,0.2),temp_tn)

plt.figure('sic-graz_inten')
plt.title('sic-graz_inten')
plt.xlabel('graz_inten(x羊/天/公顷)')
plt.ylabel('delta_sic*sic')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sic_y=np.zeros(4)
sic_y[0]=all_sic_dy_y[0:12].sum()/12.0
sic_y[1]=all_sic_dy_y[12:24].sum()/12.0
sic_y[2]=all_sic_dy_y[24:36].sum()/12.0
sic_y[3]=all_sic_dy_y[36:48].sum()/12.0
plt.scatter(soc_x,sic_y)
coef_sic = np.polyfit(soc_x, sic_y, 3)
temp_sic=[]
for i in np.arange(0,10,0.2):
    temp_sic.append(np.polyval(coef_sic, i))
plt.plot(np.arange(0,10,0.2),temp_sic)



# plt.figure('B_SM')
# plt.title('B_SM')
# B=[]
# a1=np.array([0.1802,0.7870,0.0685,0.2036,0.0808,0.1282,0.0509,0.1282,0.0808])
# b1=np.array([1.0,0.6339,0.0571,0.2887,0.5843,1.0,1.0,0,0.5000])
# SM=[]
# y_fit_lst=[]
# for i in np.arange(0,10,0.2):
#     y_fit = np.polyval(coef_soc, i)
#     y_fit_lst.append(y_fit)
#     y_fit=y_fit+16.86
#     B.append(0.5*0.5-0.3*y_fit-0.2*0.5+0.5)
#     b1[-2]=i/10.0
#     SM.append(0.6141*(a1*b1).sum())
# plt.plot(np.arange(0,10,0.2),B,label='B')
# plt.plot(np.arange(0,10,0.2),SM,label='SM')
# plt.plot(np.arange(0,10,0.2),y_fit_lst,label='y_fit')
# plt.legend()

# #绘制热力图
# r_spear=spearmanr(data_pd)[0]
# plt.figure('r_heat_map')
# plt.title('r_heat_map')
# sns.heatmap(r_spear,annot=True,xticklabels=data_pd.columns[1:],yticklabels=data_pd.columns[1:])

np.save('coef_soc.npy',coef_soc)
np.save('coef_sic.npy',coef_sic)
np.save('coef_tn.npy',coef_tn)
