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


def read_f3():
    path= 'f3_humidity.xls'
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows-1, ncols))
    title = table.row_values(0)[0:2]+table.row_values(0)[4:8]
    for i in range(nrows-1):
        rows = table.row_values(i+1)
        rows = list(map(lambda x: np.nan if x == '' else x, rows))#缺失值处理
        datamatrix[i, :] = rows

    return datamatrix[:,[0,1,4,5,6,7]],title
def read_f4():
    path= 'f4_evaporation.xls'
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows-1, ncols))
    title = table.row_values(0)[0:2]+table.row_values(0)[4:6]
    for i in range(nrows-1):
        rows = table.row_values(i+1)
        rows = list(map(lambda x: np.nan if x == '' else x, rows))
        datamatrix[i, :] = rows

    return datamatrix[:,[0,1,4,5]],title
def read_f8():
    path='f8_climate.xls'
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows-1, ncols))
    title=table.row_values(0)[4:]
    for i in range(nrows-1):
        rows = table.row_values(i+1)
        rows=list(map(lambda x: np.nan if x=='' else x,rows))
        datamatrix[i, :] = rows
    return datamatrix[:,4:], title

def read_f6_9_10():
    path= 'f6_f9_f10.xlsx'
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows-5, ncols))
    title=table.row_values(0)
    title=np.array(title)
    for i in range(nrows-5):
        rows = table.row_values(i+1)
        rows=list(map(lambda x: np.nan if x=='' else x,rows))
        datamatrix[i, :] = rows
    return datamatrix[:,[2,4,6]], title[[2,4,6]]

def outlier_process(data):
    Q1 = np.quantile(a=data, q=0.25)
    Q3 = np.quantile(a=data, q=0.75)
    QR = Q3 - Q1
    ll = Q1 - 1.5 * QR
    ul = Q3 + 1.5 * QR
    outlier_idx=np.where((data<ll)+(data>ul))[0]
    idx=np.arange(0,len(data))
    idx=np.delete(idx,outlier_idx)
    inter_f = scipy.interpolate.CubicSpline(idx, data[idx])
    data[outlier_idx]=inter_f(outlier_idx)
    return data

def get_problem2_data():
    humidity,humidity_title = read_f3() #0：月份；1：年份；2：10cm；3:40cm；4:100cm；5:200cm；
    evaporation,evaporation_title =read_f4()
    climate,climate_title =read_f8()
    f6_9_10,f6_9_10_title=read_f6_9_10()
    # np.isnan(climate).sum(0)#统计缺失值个数
    idx=np.where(np.isnan(climate).sum(0)>0)
    climate=np.delete(climate,idx,axis=1)
    climate[:,[0,1]]=climate[:,[1,0]]
    climate_title=np.delete(climate_title,idx,axis=0)
    climate_title[[0,1]]=climate_title[[1,0]]

    ##合并表
    humi_and_evap=np.hstack([humidity,evaporation[:,2:]])
    humi_and_evap_title=np.hstack([humidity_title,evaporation_title[2:]])
    humi_evap_climate=np.hstack([humi_and_evap,np.delete(climate,[3,4,5,6],axis=0)[:,2:]])
    humi_evap_climate_title=np.hstack([humi_and_evap_title,climate_title[2:]])
    humi_evap_climate=np.delete(humi_evap_climate,[6,14,23,24,25],axis=1)
    humi_evap_climate_title=np.delete(humi_evap_climate_title,[6,14,23,24,25],axis=0)
    humi_evap_climate = np.hstack([humi_evap_climate, f6_9_10])
    humi_evap_climate_title = np.hstack([humi_evap_climate_title, f6_9_10_title])

    plt.figure("Box_figure1")
    plt.title('Box_figure1')
    plt.boxplot(humi_evap_climate[:, 2:6], labels=humi_evap_climate_title[2:6])  # 绘制箱线图
    plt.figure("Box_figure2")
    plt.title('Box_figure2')
    plt.boxplot(humi_evap_climate[:, 6:10], labels=humi_evap_climate_title[6:10])  # 绘制箱线图
    plt.figure("Box_figure3")
    plt.title('Box_figure3')
    plt.boxplot(humi_evap_climate[:, 10:14], labels=humi_evap_climate_title[10:14])  # 绘制箱线图
    plt.figure("Box_figure4")
    plt.title('Box_figure4')
    plt.boxplot(humi_evap_climate[:, 14:18], labels=humi_evap_climate_title[14:18])  # 绘制箱线图
    plt.figure("Box_figure5")
    plt.title('Box_figure5')
    plt.boxplot(humi_evap_climate[:, 18:21], labels=humi_evap_climate_title[18:21])  # 绘制箱线图
    plt.figure("Box_figure6")
    plt.title('Box_figure6')
    plt.boxplot(humi_evap_climate[:, 21:], labels=humi_evap_climate_title[21:])  # 绘制箱线图
    for idx in [4,5,6,12,15,16,17,21,22,24,25]:
        humi_evap_climate[:,idx]=outlier_process(humi_evap_climate[:,idx])
    humi_evap_climate[0:2,15]=humi_evap_climate[3:5,15]
    # plt.boxplot(humi_evap_climate[:, 2:], labels=humi_evap_climate_title[2:])  # 绘制箱线图
    humi_evap_climate = humi_evap_climate[np.argsort(humi_evap_climate[:, 1] + humi_evap_climate[:, 0] / 100.0)]
    return humi_evap_climate,humi_evap_climate_title

def p2_10_pred(order=(11,0,5),seasonal_order=(3,0,2,12),if_show=False):
    data=humi_evap_climate[:, 2]
    model = SARIMAX(data,exog=humi_evap_climate[:, [6,15,25,26]], order=order, seasonal_order=seasonal_order)
    mr = model.fit(disp=False)
    pred_y = mr.predict()
    if if_show == True:
        data = data[10:]
        pred_y = pred_y[10:]
        plt.figure("pred_humi10")
        plt.plot(pred_y, label='pred_y')
        plt.plot(data, label='y')
        plt.legend()
        print("mean_absolute_error:", mean_absolute_error(pred_y, data))
        print("mean_squared_error:", mean_squared_error(pred_y, data))
        print("rmse:", sqrt(mean_squared_error(pred_y, data)))
        print("r2 score:", r2_score(pred_y, data))
    return mr

def p2_40_pred(order=(11,0,5),seasonal_order=(3,0,2,12),if_show=False):
    data=humi_evap_climate[:, 3]
    model = SARIMAX(data,exog=humi_evap_climate[:, [2,6,15,24,25]], order=order, seasonal_order=seasonal_order)
    mr = model.fit(disp=False)
    pred_y = mr.predict()
    if if_show == True:
        data = data[10:]
        pred_y = pred_y[10:]
        plt.figure("pred_humi40")
        plt.plot(pred_y, label='pred_y')
        plt.plot(data, label='y')
        plt.legend()
        print("mean_absolute_error:", mean_absolute_error(pred_y, data))
        print("mean_squared_error:", mean_squared_error(pred_y, data))
        print("rmse:", sqrt(mean_squared_error(pred_y, data)))
        print("r2 score:", r2_score(pred_y, data))
    return mr

def p2_100_pred(order=(11,0,5),seasonal_order=(3,0,2,12),if_show=False):
    data=humi_evap_climate[:, 4]
    model = SARIMAX(data,exog=humi_evap_climate[:, 3], order=order, seasonal_order=seasonal_order)
    mr = model.fit(disp=False)
    pred_y = mr.predict()
    if if_show == True:
        data = data[10:]
        pred_y = pred_y[10:]
        plt.figure("pred_humi100")
        plt.plot(pred_y, label='pred_y')
        plt.plot(data, label='y')
        plt.legend()
        print("mean_absolute_error:", mean_absolute_error(pred_y, data))
        print("mean_squared_error:", mean_squared_error(pred_y, data))
        print("rmse:", sqrt(mean_squared_error(pred_y, data)))
        print("r2 score:", r2_score(pred_y, data))
    return mr

def p2_200_pred(order=(1,0,1),seasonal_order=(1,0,1,12),if_show=False):
    data=humi_evap_climate[:, 5]
    model = SARIMAX(data,exog=humi_evap_climate[:, [3,4]], order=order, seasonal_order=seasonal_order)
    mr=model.fit(disp=False)
    pred_y = mr.predict()
    if if_show == True:
        data = data[10:]
        pred_y = pred_y[10:]
        plt.figure("pred_humi200")
        plt.plot(pred_y, label='pred_y')
        plt.plot(data, label='y')
        plt.legend()
        print("mean_absolute_error:", mean_absolute_error(pred_y, data))
        print("mean_squared_error:", mean_squared_error(pred_y, data))
        print("rmse:", sqrt(mean_squared_error(pred_y, data)))
        print("r2 score:", r2_score(pred_y, data))
    return mr

def p2_precipitation_pred(order=(11,0,5),seasonal_order=(3,0,2,12),if_show=False):
    data=humi_evap_climate[:, 15]
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order,)
    mr = model.fit(disp=False)
    pred_y = mr.predict()
    if if_show==True:
        data = data[10:]
        pred_y = pred_y[10:]
        plt.figure("pred_precipitation")
        plt.plot(pred_y, label='pred_y')
        plt.plot(data, label='y')
        plt.legend()
        print("mean_absolute_error:", mean_absolute_error(pred_y, data))
        print("mean_squared_error:", mean_squared_error(pred_y, data))
        print("rmse:", sqrt(mean_squared_error(pred_y, data)))
        print("r2 score:", r2_score(pred_y, data))
    return mr

def p2_evaporation_pred(order=(11,0,5),seasonal_order=(3,0,2,12),if_show=False):
    data=humi_evap_climate[:, 6]
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    mr = model.fit(disp=False)
    pred_y = mr.predict()
    if if_show == True:
        data = data[10:]
        pred_y = pred_y[10:]
        plt.figure("pred_evaporation")
        plt.plot(pred_y, label='pred_y')
        plt.plot(data, label='y')
        plt.legend()
        print("mean_absolute_error:", mean_absolute_error(pred_y, data))
        print("mean_squared_error:", mean_squared_error(pred_y, data))
        print("rmse:", sqrt(mean_squared_error(pred_y, data)))
        print("r2 score:", r2_score(pred_y, data))
    return mr

def p2_vor_pred(order=(11,0,5),seasonal_order=(3,0,2,12),if_show=False):
    data=humi_evap_climate[:, 24]
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    mr = model.fit(disp=False)
    pred_y = mr.predict()
    if if_show == True:
        data = data[10:]
        pred_y = pred_y[10:]
        plt.figure("pred_vor")
        plt.plot(pred_y, label='pred_y')
        plt.plot(data, label='y')
        plt.legend()
        print("mean_absolute_error:", mean_absolute_error(pred_y, data))
        print("mean_squared_error:", mean_squared_error(pred_y, data))
        print("rmse:", sqrt(mean_squared_error(pred_y, data)))
        print("r2 score:", r2_score(pred_y, data))
    return mr

def p2_ndvi_pred(order=(11,0,5),seasonal_order=(3,0,2,12),if_show=False):
    data=humi_evap_climate[:, 25]
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    mr = model.fit(disp=False)
    pred_y = mr.predict()
    if if_show == True:
        data = data[10:]
        pred_y = pred_y[10:]
        plt.figure("pred_ndvi")
        plt.plot(pred_y, label='pred_y')
        plt.plot(data, label='y')
        plt.legend()
        print("mean_absolute_error:", mean_absolute_error(pred_y, data))
        print("mean_squared_error:", mean_squared_error(pred_y, data))
        print("rmse:", sqrt(mean_squared_error(pred_y, data)))
        print("r2 score:", r2_score(pred_y, data))
    return mr

def p2_lail_pred(order=(11,0,5),seasonal_order=(3,0,2,12),if_show=False):
    data=humi_evap_climate[:, 26]
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    mr = model.fit(disp=False)
    pred_y = mr.predict()
    if if_show == True:
        data = data[10:]
        pred_y = pred_y[10:]
        plt.figure("pred_lail")
        plt.plot(pred_y, label='pred_y')
        plt.plot(data, label='y')
        plt.legend()
        print("mean_absolute_error:", mean_absolute_error(pred_y, data))
        print("mean_squared_error:", mean_squared_error(pred_y, data))
        print("rmse:", sqrt(mean_squared_error(pred_y, data)))
        print("r2 score:", r2_score(pred_y, data))
    return mr

def plot_feature_dist():
    for i in range(5):
        plt.figure("feature{}".format(i+1))
        plt.title('feature{}'.format(i+1))
        for j in range(5):
            plt.subplot(1,5,j+1)
            plt.title(humi_evap_climate_title[ 2+i*5+j])
            plt.plot(humi_evap_climate[:, 2+i*5+j])

def find_para():
    best_para=[0,0,0,0,0,0]
    best_v=100000
    num=3
    for i in range(num):
        for j in range(num):
            for k in range(num):
                for l in range(num):
                    for m in range(num):
                        for n in range(num):
                            try:
                                rmse=p2_precipitation_pred(order=(i, j, k), seasonal_order=(l, m, n, 12))
                            except:
                                continue
                            if rmse<best_v:
                                best_para=[i,j,k,l,m,n]
                                best_v=rmse
    return best_para,best_v

humi_evap_climate,humi_evap_climate_title=get_problem2_data()
np.save('humi_evap_climate.npy',humi_evap_climate)
np.save('humi_evap_climate_title.npy',humi_evap_climate_title)


# #Spearmanr相关系数
r_spear=spearmanr(humi_evap_climate[:,2:])[0]
# #绘制热力图
plt.figure('r_heat_map')
plt.title('r_heat_map')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.gcf().subplots_adjust(bottom=0.3)
sns.heatmap(r_spear,annot=True,xticklabels=humi_evap_climate_title[2:],yticklabels=humi_evap_climate_title[2:])
plot_feature_dist()
#绘制数据分布图
# sns.displot(humidity[:,2],kde=True)

if_show=True
extra_precipitation=p2_precipitation_pred(if_show=if_show).forecast(21)
extra_precipitation[0:4]=np.array([3.3,13.21,27.43,29.21])
extra_evaporation=p2_evaporation_pred(if_show=if_show).forecast(21)
extra_vor=p2_vor_pred(if_show=if_show).forecast(21)
extra_ndvi=p2_ndvi_pred(if_show=if_show).forecast(21)
extra_lail=p2_lail_pred(if_show=if_show).forecast(21)

extra_ep=np.zeros((21,4))
extra_ep[:,0]=extra_evaporation
extra_ep[:,1]=extra_precipitation
extra_ep[:,2]=extra_ndvi
extra_ep[:,3]=extra_lail
extra_10=p2_10_pred(if_show=if_show).forecast(21,exog=extra_ep)
extra_10ep=np.zeros((21,5))
extra_10ep[:,0]=extra_10
extra_10ep[:,1]=extra_evaporation
extra_10ep[:,2]=extra_precipitation
extra_10ep[:,3]=extra_vor
extra_10ep[:,4]=extra_ndvi
extra_40=p2_40_pred(if_show=if_show).forecast(21,exog=extra_10ep)
extra_100=p2_100_pred(if_show=if_show).forecast(21,exog=extra_40)
extra_40_100=np.zeros((21,2))
extra_40_100[:,0]=extra_40
extra_40_100[:,1]=extra_100
extra_200=p2_200_pred(if_show=if_show).forecast(21,exog=extra_40_100)

plt.figure('pred_humi200')
plt.plot(list(range(113,134,1)),extra_200,label='pred_future')
plt.legend()

plt.figure('pred_humi100')
plt.plot(list(range(113,134,1)),extra_100,label='pred_future')
plt.legend()

plt.figure('pred_humi40')
plt.plot(list(range(113,134,1)),extra_40,label='pred_future')
plt.legend()

plt.figure('pred_humi10')
plt.plot(list(range(113,134,1)),extra_10,label='pred_future')
plt.legend()

plt.figure('pred_ndvi')
plt.plot(list(range(113,134,1)),extra_ndvi,label='pred_future')
plt.legend()

plt.figure('pred_lail')
plt.plot(list(range(113,134,1)),extra_lail,label='pred_future')
plt.legend()

plt.figure('pred_evaporation')
plt.plot(list(range(113,134,1)),extra_evaporation,label='pred_future')
plt.legend()

plt.figure('pred_precipitation')
plt.plot(list(range(113,134,1)),extra_precipitation,label='pred_future')
plt.legend()

plt.figure('pred_vor')
plt.plot(list(range(113,134,1)),extra_vor,label='pred_future')
plt.legend()

np.save('extra_10.npy',extra_10)
np.save('extra_40.npy',extra_40)
np.save('extra_100.npy',extra_100)
np.save('extra_200.npy',extra_200)
