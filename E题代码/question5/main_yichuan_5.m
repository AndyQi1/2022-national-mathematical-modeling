%%遗传算法求解第五问最优值
clc;clear
global rf
global inc
global pop
global humidity
humi = 16.13;
rf=0.6339;
pop=1.0;
inc=1.0;
a1 = [0.1802, 0.07870, 0.0685, 0.2036, 0.0808, 0.1282, 0.0509, 0.1282, 0.0808];
b1 = [1.0, rf, 0.0571, 0.2887, 0.5843, 1.0, pop, 0, inc];
rain_fall_lst=[300,600,900,1200];
rain_Q = [1.0000,0.7704,0.2296,0];
population_4 = [0.83,1,0.5,0.47];
income_4= [0.2868,0.6953,1.0000,0.0669];
% delta_humi_lst=np.polyval(coef_hr, rainfall_lst)
best_res = zeros(16);
fval_res = zeros(16);
options = gaoptimset('PopulationSize', 300);
nvars = 1;  A = [];  b = [];
Aeq = []; beq = []; lb = 0; ub = 10;
ceq = [];
for i=1:4
    delta_humi_lst = 8.17159639e-08*rain_fall_lst(i)^3 - 1.43417031e-04*rain_fall_lst(i)^2 + 7.70188794e-02*rain_fall_lst(i) - 1.23636167e+01;
    humidity = humi + delta_humi_lst;
    rf = rain_Q(i);
    for j = 1:4
        pop = population_4(j);
        inc = income_4(j);
        nonlcon = @nonlconfun;
        [x_best,fval] = ga(@fun5, nvars, A, b, Aeq, beq, lb, ub, nonlcon);
        best_res((i-1)*4+j,1) = x_best;
        fval_res((i-1)*4+j,1) = fval;
    end
end
