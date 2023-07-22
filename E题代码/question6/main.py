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
from pyswarm import pso


def delta_w_for_graze(w, s):
    return 0.049*w*(1-w/4000.0)-0.0047*s*w

coef_plants=np.load('coef_plants.npy')
coef_soc=np.load('coef_soc.npy')
coef_sic=np.load('coef_sic.npy')
coef_tn=np.load('coef_tn.npy')

graz_inten_4=np.array([[275.86,272.22,256.39],[230.0,210,200],[601.75,492,520.75],[245.9,245.9,255.7]])
graz_inten_4=graz_inten_4.mean(1)/100
best_inten=np.array([3.8,3.8,3.8,3.8])
plants_4=np.zeros((2,4,4))
plants_4[0,0,:]=np.array([124.2,92.81,100.16,61.78])
plants_4[1,0,:]=np.array([124.2,92.81,100.16,61.78])
soc=np.zeros((2,4,4))
soc[0,0,:]=np.array([16.76,16.76,14.65,16.76])
soc[1,0,:]=np.array([16.76,16.76,14.65,16.76])
tn=np.zeros((2,4,4))
tn[0,0,:]=np.array([2.07,2.07,1.87,2.07])
tn[1,0,:]=np.array([2.07,2.07,1.87,2.07])

time=np.array([2020,2021,2022,2023])
for i in range(3):
    delta_w=delta_w_for_graze(plants_4[0,i,:],graz_inten_4)
    plants_4[0,i+1,:]=plants_4[0,i,:]+delta_w

    delta_w = delta_w_for_graze(plants_4[1, i, :], best_inten)
    plants_4[1, i + 1, :] = plants_4[1, i, :] + delta_w

    delta_soc=np.polyval(coef_soc,graz_inten_4)/soc[0,i,:]/2.0
    soc[0,i+1,:]=soc[0,i,:]+delta_soc

    delta_soc = np.polyval(coef_soc, best_inten) / soc[1, i, :] / 2.0
    soc[1, i + 1, :] = soc[1, i, :] + delta_soc


    delta_tn = np.polyval(coef_tn, graz_inten_4) / tn[0,i, :]/2.0
    tn[0,i + 1, :] = tn[0,i, :] + delta_tn

    delta_tn = np.polyval(coef_tn, best_inten) / tn[1, i, :] / 2.0
    tn[1, i + 1, :] = tn[1, i, :] + delta_tn



figure_lst=['plants_num','SOC','TN']
data_lst=[plants_4,soc,tn]
x_lst=['2020','2021','2022','2023']
y_lable_lst=['g/m^2','g/kg','g/kg']
for i in range(3):
    plt.figure(figure_lst[i])
    plt.subplot(1,2,1)
    plt.title(figure_lst[i]+'_cur_policy')
    plt.xlabel('time(year)')
    plt.ylabel(y_lable_lst[i])
    for j in range(4):
        plt.plot(x_lst,data_lst[i][0,:,j],label='farmer_{}'.format(j+1))
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title(figure_lst[i] + '_best_policy')
    plt.xlabel('time(year)')
    plt.ylabel(y_lable_lst[i])
    for j in range(4):
        plt.plot(x_lst, data_lst[i][1,:, j], label='farmer_{}'.format(j + 1))
    plt.legend()





