%%遗传算法求解第四问最优值
clc;clear
options = gaoptimset('PopulationSize', 100);
nvars = 1;  A = [];  b = [];
Aeq = []; beq = []; lb = 0; ub = 10;
nonlcon = [];
[x_best,fval] = ga(@fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, options)
