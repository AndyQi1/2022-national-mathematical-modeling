function [c,ceq] = nonlconfun(x)
global rf
global inc
global pop
global humidity
a1 = [0.1802, 0.07870, 0.0685, 0.2036, 0.0808, 0.1282, 0.0509, 0.1282, 0.0808];
b1 = [1.0, rf, 0.0571, 0.2887, 0.5843, 1.0, pop, 0, inc];
b1(8) = x(1) / 10.0;
p=2*(0.5-b1(8))^2-0.2;
income=1/(1-b1(8)+0.1)*inc;
b1(9)=income;
b1(4)=p;
SM = 0.5 * a1*b1';
delta_organic_content = 0.01567787*x(1)^3 - 0.14720232*x(1)^2 + 0.15082857*x(1) + 1.11696183;
organic_content = delta_organic_content + 16.86;
volume_weight = (0.4/3.42)*delta_organic_content+1.4 - 1;
B = 0.6232*volume_weight^3-0.2395*(organic_content-10)*1.724/20.0/1.724-0.1373*(humidity-10)/60+0.3768;
c = [B-0.35;SM-0.29];
ceq = [];
end