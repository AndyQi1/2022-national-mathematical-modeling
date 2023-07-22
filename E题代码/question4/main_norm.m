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


% 因子强度
Q = zeros(Qnum,year);
element = [wind;rain;temperature;plantation;surfacewater;waterinfrastructure;...
    population;livestock;income]; % 1风速，2降雨，3温度，4植被，5地表水，6水文设施，7人口，8畜牧，9收入
for t = 1:Qnum
    Q(t,:) = element(t,:)/max(element(t,:));
end
Q([2 3 4 5 6],:) = 1 - Q([2 3 4 5 6],:);
eta = 0.5;
SM = eta*W*Q;
figure
plot(2016:2021,SM,'b-o');
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
Q_common = [muhu_wind;muhu_rain;muhu_temperature;...
    muhu_plantation;muhu_surfacewater;muhu_waterinfrastructure];
for k = 1:6
    Q_common(k,:) = Q_common(k,:)/max(Q_common(k,:));
end

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
% 人文因素尺度变换
muhu_population = [muhu1_population,muhu2_population,muhu3_population,muhu4_population];
muhu_livestock = [muhu1_livestock,muhu2_livestock,muhu3_livestock,muhu4_livestock];
muhu_income = [muhu1_income,muhu2_income,muhu3_income,muhu4_income];
Q_population = muhu_population/max(muhu_population);
Q_livestock = muhu_livestock/max(muhu_livestock);
Q_income = muhu_income/max(muhu_income);
% 牧户的因素
Q1 = [Q_common;Q_population(1:3);Q_livestock(1:3);Q_income(1:3)];
Q2 = [Q_common;Q_population(4:6);Q_livestock(4:6);Q_income(4:6)];
Q3 = [Q_common;Q_population(7:9);Q_livestock(7:9);Q_income(7:9)];
Q4 = [Q_common;Q_population(10:12);Q_livestock(10:12);Q_income(10:12)];

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
muhu_livestock_2018 = [muhu1_livestock(1),muhu2_livestock(1),...
    muhu3_livestock(1),muhu4_livestock(1)];
[muhu_livestock_2018,Index] = sort(muhu_livestock_2018);
SM_2018 = [SM1(1),SM2(1),SM3(1),SM4(1)];
figure
plot(muhu_livestock_2018,SM_2018(Index),'bo-');grid on;
xlabel('放牧强度（羊单位/km^2）');
ylabel('沙漠化程度');
% 2019年
muhu_livestock_2019 = [muhu1_livestock(2),muhu2_livestock(2),...
    muhu3_livestock(2),muhu4_livestock(2)];
[muhu_livestock_2019,Index] = sort(muhu_livestock_2019);
SM_2019 = [SM1(2),SM2(2),SM3(2),SM4(2)];
figure
plot(muhu_livestock_2019,SM_2019(Index),'bo-');grid on;
xlabel('放牧强度（羊单位/km^2）');
ylabel('沙漠化程度');
% 2020年
muhu_livestock_2020 = [muhu1_livestock(3),muhu2_livestock(3),...
    muhu3_livestock(3),muhu4_livestock(3)];
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
rain_pre = 300*[1:4];
rain_Q = rain_pre/rain_pre(end);

Q11 = repmat(Q1(:,2),[1,4]);
Q21 = repmat(Q2(:,2),[1,4]);
Q31 = repmat(Q3(:,2),[1,4]);
Q41 = repmat(Q4(:,2),[1,4]);
Q11(2,:) = rain_Q;
Q21(2,:) = rain_Q;
Q31(2,:) = rain_Q;
Q41(2,:) = rain_Q;
SM11 = eta*W*Q11;
SM21 = eta*W*Q21;
SM31 = eta*W*Q31;
SM41 = eta*W*Q41;
figure
plot(300*[1:4],SM11,300*[1:4],SM21,300*[1:4],SM31,300*[1:4],SM41);