%% 2022华为杯E题——第四问
close all;clc;clear all;
Qnum = 9; % 因子个数
year = 6; % 2016-2021
W = [0.1802,0.0787,0.0685,0.2036,0.0808,0.1282,0.0509,0.1282,0.0808];
% 年鉴数据 2016-2021
wind = [3.1,3.2,3.2,3.3,3.2,3.2]; % m/s
rain = [412.8,309,168.7,277.6,293.5,389.7]; % mm
temperature = [3.9,3.5,4.7,4.3,4.4,3.3]; % ^0C
plantation = [0.953,0.953,0.886,0.954,0.954,0.954]; % %
surfacewater = [1.724e4,1.639e4,1.641e4,1.641e4,1.641e4,1.601e4]/1e4; % m^3/km^2
waterinfrastructure = [4866,4866,4866,4866,738,738]; % km^2
% groundwater = [0,0,0,0,0,1.203e4]; % m^3/km^2
population = [5.146,5.167,5.191,5.206,5.224,5.473]; % person/km^2
livestock = [116.39,124.02,111.11,105.62,109.52,113.57]; % unit/km^2
% economy = [2334.75,2554.75,2761.5,3226.9,3494.7,3939.75]; % yuan/month
income = [406,614,297,1404,3700,8735]; % yuan/year
% 判断阈值
rho = 1; % 系数
wind_thre = mean(wind) + rho*sqrt(var(wind))*[-1 1];
rain_thre = mean(rain) + rho*sqrt(var(rain))*[-1 1];
temperature_thre = mean(temperature) + rho*sqrt(var(temperature))*[-1 1];

plantation_thre = mean(plantation) + rho*sqrt(var(plantation))*[-1 1];
surfacewater_thre = mean(surfacewater) + rho*sqrt(var(surfacewater))*[-1 1];
waterinfrastructure_thre = mean(waterinfrastructure) + rho*sqrt(var(waterinfrastructure))*[-1 1];
% groundwater_thre = mean(groundwater) + rho*var(groundwater)*[-1 1];

population_thre = mean(population) + rho*sqrt(var(population))*[-1 1];
livestock_thre = mean(livestock) + rho*sqrt(var(livestock))*[-1 1];
% economy_thre = mean(economy) + rho*var(economy)*[-1 1];
income_thre = mean(income) + rho*sqrt(var(income))*[-1 1];
% 因子强度
Q = zeros(Qnum,year);
element = [wind;rain;temperature;plantation;surfacewater;waterinfrastructure;...
    population;livestock;income]; % 1风速，2降雨，3温度，4植被，5地表水，6水文设施，7人口，8畜牧，9收入
threshold = [wind_thre;rain_thre;temperature_thre;...
    plantation_thre;surfacewater_thre;waterinfrastructure_thre;...
    population_thre;livestock_thre;income_thre];
for t = [1 7 8 9] % 正比关系
    for k = 1:year
    if element(t,k) < threshold(t,1)
        Q(t,k) = 0;
    elseif element(t,k) >= threshold(t,2)
        Q(t,k) = 1;
    else
        Q(t,k) = (element(t,k) - threshold(t,1))/(threshold(t,2)-threshold(t,1));
    end
    end
end
for t = [2 3 4 5 6] % 反比关系
    for k = 1:year
    if element(t,k) < threshold(t,1)
        Q(t,k) = 1;
    elseif element(t,k) >= threshold(t,2)
        Q(t,k) = 0;
    else
        Q(t,k) = (threshold(t,2) - element(t,k))/(threshold(t,2)-threshold(t,1));
    end
    end
end
SM = W*Q;
eta = 0.75;
modifiedSM = eta*SM;
figure
plot(2016:2021,modifiedSM,'b-o');
xticks([2016:2021]);
xlabel('年份');
ylabel('沙漠化程度');
grid on;
%% 2018-2020
muhu_wind = [3.33,3.63,3.42];
muhu_rain = [206.37,322.85,285.15];
muhu_temperature = [4.78,4.32,4.5];
muhu_plantation = [0.886,0.954,0.954];
muhu_surfacewater = [1.641e4,1.641e4,1.641e4]/1e4;
muhu_waterinfrastructure = [4866,4866,738];

muhu1_population = [0.83 0.83 0.83];
muhu1_livestock = [275.83 272.22 256.39];
muhu1_income = [10.56 15.54 12.86];

muhu2_population = [1 1 1];
muhu2_livestock = [230 210 200];
muhu2_income = [21.95 24.195 31.175];

muhu3_population = [0.5 0.5 0.5];
muhu3_livestock = [601.75 492 520.75];
muhu3_income = [38.925 42.3375 44.8375];

muhu4_population = [0.47 0.47 0.47];
muhu4_livestock = [245.9 245.9 255.7];
muhu4_income = [3.33 3.33 3];
% 存储实际的放牧压力，防止画图出现负值
muhu1_livestock1 = [275.83 272.22 256.39];
muhu2_livestock1 = [230 210 200];
muhu3_livestock1 = [601.75 492 520.75];
muhu4_livestock1 = [245.9 245.9 255.7];
% 人文因素尺度变换
muhu_population = [muhu1_population,muhu2_population,muhu3_population,muhu4_population];
muhu_livestock = [muhu1_livestock,muhu2_livestock,muhu3_livestock,muhu4_livestock];
muhu_income = [muhu1_income,muhu2_income,muhu3_income,muhu4_income];
muhu_population_mean = mean(muhu_population);
muhu_population_stdvar = sqrt(var(muhu_population));
muhu_livestock_mean = mean(muhu_livestock);
muhu_livestock_stdvar = sqrt(var(muhu_livestock));
muhu_income_mean = mean(muhu_income);
muhu_income_stdvar = sqrt(var(muhu_income));

muhu1_population = mean(population) + (muhu1_population - muhu_population_mean)...
    *muhu_population_stdvar/sqrt(var(population));
muhu2_population = mean(population) + (muhu2_population - muhu_population_mean)...
    *muhu_population_stdvar/sqrt(var(population));
muhu3_population = mean(population) + (muhu3_population - muhu_population_mean)...
    *muhu_population_stdvar/sqrt(var(population));
muhu4_population = mean(population) + (muhu4_population - muhu_population_mean)...
    *muhu_population_stdvar/sqrt(var(population));

muhu1_livestock = mean(livestock) + (muhu1_livestock - muhu_livestock_mean)...
    *muhu_livestock_stdvar/sqrt(var(livestock));
muhu2_livestock = mean(livestock) + (muhu2_livestock - muhu_livestock_mean)...
    *muhu_livestock_stdvar/sqrt(var(livestock));
muhu3_livestock = mean(livestock) + (muhu3_livestock - muhu_livestock_mean)...
    *muhu_livestock_stdvar/sqrt(var(livestock));
muhu4_livestock = mean(livestock) + (muhu4_livestock - muhu_livestock_mean)...
    *muhu_livestock_stdvar/sqrt(var(livestock));

muhu1_income = mean(income) + (muhu1_income - muhu_income_mean)...
    *muhu_income_stdvar/sqrt(var(income));
muhu2_income = mean(income) + (muhu2_income - muhu_income_mean)...
    *muhu_income_stdvar/sqrt(var(income));
muhu3_income = mean(income) + (muhu3_income - muhu_income_mean)...
    *muhu_income_stdvar/sqrt(var(income));
muhu4_income = mean(income) + (muhu4_income - muhu_income_mean)...
    *muhu_income_stdvar/sqrt(var(income));
% 牧户的因素
muhu1 = [muhu_wind;muhu_rain;muhu_temperature;...
    muhu_plantation;muhu_surfacewater;muhu_waterinfrastructure;...
    muhu1_population;muhu1_livestock;muhu1_income];
muhu2 = [muhu_wind;muhu_rain;muhu_temperature;...
    muhu_plantation;muhu_surfacewater;muhu_waterinfrastructure;...
    muhu2_population;muhu2_livestock;muhu2_income];
muhu3 = [muhu_wind;muhu_rain;muhu_temperature;...
    muhu_plantation;muhu_surfacewater;muhu_waterinfrastructure;...
    muhu3_population;muhu3_livestock;muhu3_income];
muhu4 = [muhu_wind;muhu_rain;muhu_temperature;...
    muhu_plantation;muhu_surfacewater;muhu_waterinfrastructure;...
    muhu4_population;muhu4_livestock;muhu4_income];
%% 计算各个牧户的因子强度
muhu_year = 3;
Q1 = zeros(Qnum,muhu_year);Q2 = zeros(Qnum,muhu_year);
Q3 = zeros(Qnum,muhu_year);Q4 = zeros(Qnum,muhu_year);
% 牧户1
for t = [1 7 8 9] % 正比关系
    for k = 1:muhu_year
    if muhu1(t,k) < threshold(t,1)
        Q1(t,k) = 0;
    elseif muhu1(t,k) >= threshold(t,2)
        Q1(t,k) = 1;
    else
        Q1(t,k) = (muhu1(t,k) - threshold(t,1))/(threshold(t,2)-threshold(t,1));
    end
    end
end
for t = [2 3 4 5 6] % 反比关系
    for k = 1:muhu_year
    if muhu1(t,k) < threshold(t,1)
        Q1(t,k) = 1;
    elseif muhu1(t,k) >= threshold(t,2)
        Q1(t,k) = 0;
    else
        Q1(t,k) = (threshold(t,2) - muhu1(t,k))/(threshold(t,2)-threshold(t,1));
    end
    end
end
% 牧户2
for t = [1 7 8 9] % 正比关系
    for k = 1:muhu_year
    if muhu2(t,k) < threshold(t,1)
        Q2(t,k) = 0;
    elseif muhu2(t,k) >= threshold(t,2)
        Q2(t,k) = 1;
    else
        Q2(t,k) = (muhu2(t,k) - threshold(t,1))/(threshold(t,2)-threshold(t,1));
    end
    end
end
for t = [2 3 4 5 6] % 反比关系
    for k = 1:muhu_year
    if muhu2(t,k) < threshold(t,1)
        Q2(t,k) = 1;
    elseif muhu2(t,k) >= threshold(t,2)
        Q2(t,k) = 0;
    else
        Q2(t,k) = (threshold(t,2) - muhu2(t,k))/(threshold(t,2)-threshold(t,1));
    end
    end
end
% 牧户3
for t = [1 7 8 9] % 正比关系
    for k = 1:muhu_year
    if muhu3(t,k) < threshold(t,1)
        Q3(t,k) = 0;
    elseif muhu3(t,k) >= threshold(t,2)
        Q3(t,k) = 1;
    else
        Q3(t,k) = (muhu3(t,k) - threshold(t,1))/(threshold(t,2)-threshold(t,1));
    end
    end
end
for t = [2 3 4 5 6] % 反比关系
    for k = 1:muhu_year
    if muhu3(t,k) < threshold(t,1)
        Q3(t,k) = 1;
    elseif muhu3(t,k) >= threshold(t,2)
        Q3(t,k) = 0;
    else
        Q3(t,k) = (threshold(t,2) - muhu3(t,k))/(threshold(t,2)-threshold(t,1));
    end
    end
end
% 牧户4
for t = [1 7 8 9] % 正比关系
    for k = 1:muhu_year
    if muhu4(t,k) < threshold(t,1)
        Q4(t,k) = 0;
    elseif muhu4(t,k) >= threshold(t,2)
        Q4(t,k) = 1;
    else
        Q4(t,k) = (muhu4(t,k) - threshold(t,1))/(threshold(t,2)-threshold(t,1));
    end
    end
end
for t = [2 3 4 5 6] % 反比关系
    for k = 1:muhu_year
    if muhu4(t,k) < threshold(t,1)
        Q4(t,k) = 1;
    elseif muhu4(t,k) >= threshold(t,2)
        Q4(t,k) = 0;
    else
        Q4(t,k) = (threshold(t,2) - muhu4(t,k))/(threshold(t,2)-threshold(t,1));
    end
    end
end
SM1 = eta*W*Q1;
SM2 = eta*W*Q2;
SM3 = eta*W*Q3;
SM4 = eta*W*Q4;
figure
plot(2018:2020,SM1,'b-o',2018:2020,SM2,'r-*',...
    2018:2020,SM3,'k-^',2018:2020,SM4,'g-s');
xticks([2018:2020]);grid on;
legend('牧户1','牧户2','牧户3','牧户4');
xlabel('年份');
ylabel('沙漠化程度');
% 2018年
muhu_livestock_2018 = [muhu1_livestock1(1),muhu2_livestock1(1),...
    muhu3_livestock1(1),muhu4_livestock1(1)];
[muhu_livestock_2018,Index] = sort(muhu_livestock_2018);
SM_2018 = [SM1(1),SM2(1),SM3(1),SM4(1)];
figure
plot(muhu_livestock_2018,SM_2018(Index),'bo-');grid on;
xlabel('放牧强度（羊单位/km^2）');
ylabel('沙漠化程度');
% 2019年
muhu_livestock_2019 = [muhu1_livestock1(2),muhu2_livestock1(2),...
    muhu3_livestock1(2),muhu4_livestock1(2)];
[muhu_livestock_2019,Index] = sort(muhu_livestock_2019);
SM_2019 = [SM1(2),SM2(2),SM3(2),SM4(2)];
figure
plot(muhu_livestock_2019,SM_2019(Index),'bo-');grid on;
xlabel('放牧强度（羊单位/km^2）');
ylabel('沙漠化程度');
% 2020年
muhu_livestock_2020 = [muhu1_livestock1(3),muhu2_livestock1(3),...
    muhu3_livestock1(3),muhu4_livestock1(3)];
[muhu_livestock_2020,Index] = sort(muhu_livestock_2020);
SM_2020 = [SM1(3),SM2(3),SM3(3),SM4(3)];
figure
plot(muhu_livestock_2020,SM_2020(Index),'bo-');grid on;
xlabel('放牧强度（羊单位/km^2）');
ylabel('沙漠化程度');
% % 总共
% muhu_livestock_total = [muhu1_livestock1,muhu2_livestock1,...
%     muhu3_livestock1,muhu4_livestock1];
% [muhu_livestock_total,Index] = sort(muhu_livestock_total);
% SM_total = [SM1,SM2,SM3,SM4];
% figure
% plot(muhu_livestock_total,SM_total(Index),'bo-');grid on;
% xlabel('放牧压力（羊单位/km^2）');
% ylabel('沙漠化程度');
%% 不同降水量
rain_pre = [200,280,360,440];
rain_pre_mean = mean(rain_pre);
rain_pre_stdvar = sqrt(var(rain_pre));
rain_pre = mean(rain) + (rain_pre - rain_pre_mean)...
    *rain_pre_stdvar/sqrt(var(rain));
rain_Q = zeros(1,length(rain_pre));
for t = 1:length(rain_pre)
    if rain_pre(t) < threshold(2,1)
        rain_Q(t) = 1;
    elseif rain_pre(t) >= threshold(2,2)
        rain_Q(t) = 0;
    else
        rain_Q(t) = (threshold(2,2) - rain_pre(t))/(threshold(2,2)-threshold(2,1));
    end
end
Q11 = repmat(Q1(:,2),[1,4]);
Q21 = repmat(Q2(:,2),[1,4]);
Q31 = repmat(Q3(:,2),[1,4]);
Q41 = repmat(Q4(:,2),[1,4]);
Q11(2,:) = rain_Q;
Q21(2,:) = rain_Q;
Q31(2,:) = rain_Q;
Q41(2,:) = rain_Q;
SM11 = W*Q11;
SM21 = W*Q21;
SM31 = W*Q31;
SM41 = W*Q41;
figure
plot(300*[1:4],SM11,300*[1:4],SM21,300*[1:4],SM31,300*[1:4],SM41);