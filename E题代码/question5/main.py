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

def compute_SM(graz_inten,rf=0.6339,pop=1.0,inc=1.0):
    a1 = np.array([0.1802, 0.07870, 0.0685, 0.2036, 0.0808, 0.1282, 0.0509, 0.1282, 0.0808])
    b1 = np.array([1.0, rf, 0.0571, 0.2887, 0.5843, 1.0, pop, 0, inc])
    b1[-2] = graz_inten / 10.0
    p=2*(0.5-b1[-2])**2-0.2
    income=1/(1-b1[-2]+0.1)*inc
    b1[8]=income
    b1[3]=p
    return 0.5 * (a1 * b1).sum()

def compute_B(graz_inten,humi=16.13):
    humidity=humi  #40.34
    delta_organic_content = np.polyval(coef_soc, graz_inten)
    organic_content = delta_organic_content + 16.86
    volume_weight = (0.4/3.42)*delta_organic_content+1.4 - 1
    B=0.6232*volume_weight**3-0.2395*(organic_content-10)*1.724/20.0/1.724-0.1373*(humidity-10)/60+0.3768
    return B

def pso_method(sm_t=0.29,b_t=0.35,rf=0.6339,pop=1.0,inc=1.0,humi=16.13):
    def target_func(x):
        return -x

    def con(x):
        SM_t=sm_t
        B_t=b_t
        return [SM_t-compute_SM(x,rf=rf,pop=pop,inc=inc),B_t-compute_B(x,humi=humi)]
    lb = [0]
    ub = [10]
    xopt, fopt = pso(target_func, lb, ub,f_ieqcons=con,maxiter=1000)
    return xopt, fopt

coef_soc=np.load('coef_soc.npy')
coef_hr=np.load('coef_hr.npy')
rainfall_lst=[300,600,900,1200]
rain_Q = [1.0000,0.7704,0.2296,0]
population_4 = [0.83,1,0.5,0.47]
income_4= [0.2868,0.6953,1.0000,0.0669]
delta_humi_lst=np.polyval(coef_hr, rainfall_lst)
best_graz_inten=np.zeros((4,4))
for i in range(4):
    temp_rf_coef=rain_Q[i]
    temp_rf=rainfall_lst[i]
    temp_delta_humi=delta_humi_lst[i]
    plt.figure(i+1)
    plt.title('rainfall={}'.format(temp_rf))
    for j in range(4):
        temp_pop_coef=population_4[j]
        temp_inc_coef=income_4[j]
        B = []
        SM = []
        y_fit_lst = []
        for k in np.arange(0, 10, 0.2):
            B.append(compute_B(k,humi=16.13+temp_delta_humi))
            SM.append(compute_SM(k,rf=temp_rf_coef,pop=temp_pop_coef,inc=temp_inc_coef))
        plt.subplot(2,2,j+1)
        plt.title('farmer_{}(rf={})-'.format(j+1,temp_rf)+'B_SM--graz_inten')
        plt.xlabel('graz_inten(x羊/天/公顷)')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.plot(np.arange(0, 10, 0.2), B, label='B')
        plt.plot(np.arange(0, 10, 0.2), SM, label='SM')
        # plt.plot(np.arange(0,10,0.2),y_fit_lst,label='y_fit')
        best_graz_inten[i,j],_=pso_method(rf=temp_rf_coef,pop=temp_pop_coef,inc=temp_inc_coef,humi=16.13+temp_delta_humi)
        plt.vlines([best_graz_inten[i,j]], 0.25, 0.6, linestyles='dashed', colors='red')
        plt.text(best_graz_inten[i,j]-0.05, 0.5, 'max_num:{:.3f}羊/天/公顷'.format(best_graz_inten[i,j]), ha='right', fontsize=10, c='red')
        plt.legend()





