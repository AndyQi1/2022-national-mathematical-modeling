import numpy as np
import matplotlib.pyplot as plt


humi_evap_climate=np.load('humi_evap_climate.npy')
humi_evap_climate_title=np.load('humi_evap_climate_title.npy')
humi_evap_climate_title=humi_evap_climate_title[[0,1,2,3,4,15]]
humi_evap_climate=humi_evap_climate[:,[0,1,2,3,4,15]]
data_year=np.zeros((10,4))
for i in range(10):
    year=humi_evap_climate[12*i,1]
    for j in range(4):
        data_year[i,j]=humi_evap_climate[np.where(humi_evap_climate[:,1]==year)[0],j+2].mean()
data_year[:,3]=data_year[:,3]*12
avg_data_year=data_year[:,0:3].sum(1)/3
y=data_year[:,0]
y=y[1:]-y[0:-1]
x=data_year[1:,3]


plt.figure('humi-rainfall')
plt.title('humi-rainfall')
plt.xlabel('rainfall')
plt.ylabel('delta_humi')
coef_hr = np.polyfit(x, y, 3)
plt.scatter(x,y)
temp_hr=[]
for i in np.arange(200,1201,100):
    temp_hr.append(np.polyval(coef_hr, i))
plt.plot(np.arange(200,1201,100),temp_hr)
np.save('coef_hr.npy',coef_hr)