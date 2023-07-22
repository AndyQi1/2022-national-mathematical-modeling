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

data=np.load('humi_evap_climate.npy')
data_title=np.load('humi_evap_climate_title.npy')
extra_10=np.load('extra_10.npy')
extra_40=np.load('extra_40.npy')
tempe=[]
humi10=[]
humi40=[]
for i in [2012.0,2014.0,2016.0,2018.0,2020.0]:
    tempe.append(data[np.where(data[:, 1] == i)[0], 7].mean())
    humi10.append(data[np.where(data[:, 1] == i)[0], 2].mean())
    humi40.append(data[np.where(data[:, 1] == i)[0], 3].mean())
model = SARIMAX(data[:,7], order=(11,0,5),seasonal_order=(3,0,2,12))
mr = model.fit(disp=False)
pred_y = mr.predict()
pred_y = pred_y[10:]
plt.figure("pred_temperature")
plt.title("pred_temperature")
plt.plot(pred_y, label='pred_y')
plt.plot(data[:,7][10:], label='y')
plt.legend()
plt.plot(list(range(113,122,1)),mr.forecast(9),label='pred_future')
plt.legend()
print("rmse:", sqrt(mean_squared_error(pred_y, data[:,7][10:])))
tempe.append(np.hstack((mr.forecast(9),data[np.where(data[:, 1] == 2022.0)[0],7])).mean())
humi10.append(np.hstack((extra_10[:9],data[np.where(data[:, 1] == 2022.0)[0],2])).mean())
humi40.append(np.hstack((extra_40[:9],data[np.where(data[:, 1] == 2022.0)[0],3])).mean())
pred_2022_x=np.array([1,tempe[-1],humi10[-1],humi40[-1]])
np.save('pred_2022_x.npy',pred_2022_x)

