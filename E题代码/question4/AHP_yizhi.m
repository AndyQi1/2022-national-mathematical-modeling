%%土壤板结化权重因子一致性检验
disp('请输入判断矩阵A')
A=[1,3,4;1/3,1,2;1/4,1/2,1];
[n,n] = size(A);
[V,D] = eig(A);%求出矩阵A的特征值和特征向量
Max_eig = max(max(D));%找到矩阵A的最大特征值
% 下面是计算一致性比例CR的环节 % 
CI = (Max_eig - n) / (n-1);
RI=[0 0.0001 0.52 0.89 1.12 1.26 1.36 1.41 1.46 1.49 1.52 1.54 1.56 1.58 1.59];
%注意哦，这里的RI最多支持 n = 15
% 这里n=2时，一定是一致矩阵，所以CI = 0，我们为了避免分母为0，将这里的第二个元素改为了很接近0的正数
CR=CI/RI(n);
disp('一致性指标CI=');disp(CI);
disp('一致性比例CR=');disp(CR);
if CR<0.10
    disp('因为CR<0.10，所以该判断矩阵A的一致性可以接受!');
else
    disp('注意：CR >= 0.10，因此该判断矩阵A需要进行修改!');
end
