import numpy as np
import matplotlib.pyplot as plt
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

coef_soc=np.load('coef_soc.npy')
coef_sic=np.load('coef_sic.npy')
coef_tn=np.load('coef_tn.npy')
plt.figure('B_SM--graz_inten')
plt.title('B_SM--graz_inten')
plt.xlabel('graz_inten(x羊/天/公顷)')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
B=[]
SM=[]
y_fit_lst=[]
for i in np.arange(0,10,0.2):
    B.append(compute_B(i))
    SM.append(compute_SM(i))
plt.plot(np.arange(0,10,0.2),B,label='B')
plt.plot(np.arange(0,10,0.2),SM,label='SM')
# plt.plot(np.arange(0,10,0.2),y_fit_lst,label='y_fit')
plt.legend()



def pso_method():
    def target_func(x):
        return compute_SM(x)+compute_B(x)
    lb = [0]
    ub = [10]
    xopt, fopt = pso(target_func, lb, ub)
    return xopt, fopt

best_graz_intensity,min_SM_B=pso_method()
GA_best_value=3.80
plt.vlines([best_graz_intensity[0],GA_best_value], 0.25, 0.6, linestyles='dashed', colors='red')
plt.text(best_graz_intensity-0.05, 0.4, 'best_value_pso:{:.3f}'.format(best_graz_intensity[0]), ha='right', fontsize=10, c='red')
plt.text(GA_best_value-0.05, 0.5, 'best_value_ga:{:.3f}'.format(GA_best_value), ha='right', fontsize=10, c='red')
plt.savefig('B_SM--graz_inten.png')