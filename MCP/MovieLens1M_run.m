clear; clc;
clear; clc;
addpath('tools');
addpath('data');

load('movielens1m.mat');

para.data = 'movielens1m';

rng('default');
rng(30); 

[row, col, val] = find(data);

[m, n] = size(data);


clear user item;


val = val - mean(val);
val = val/std(val);

idx = randperm(length(val));

train_Idx = idx(1:floor(length(val)*0.7));
test_Idx = idx(ceil(length(val)*0.3): end);

clear idx;

train_Data = sparse(row(train_Idx), col(train_Idx), val(train_Idx));
train_Data(size(data,1), size(data,2)) = 0;

para.test.row  = row(test_Idx);
para.test.col  = col(test_Idx);
para.test.data = val(test_Idx);
para.test.m = m;
para.test.n = n;

clear m n;
clear data;

theta = 5;

para.maxR = 5;
para.maxtime = 20;

para.regType = 4;
para.maxIter = 20000;
lambda = 0.1;
para.fun_num = 4;



para.tol = 1e-9;

[m, n] = size(train_Data);


R = randn(n, para.maxR);
para.R = R;
clear m n;
U0 = powerMethod( train_Data, R, para.maxR, 1e-6);
[~, ~, V0] = svd(U0'*train_Data, 'econ');
para.U0 = U0;
para.V0 = V0;

para.fun_num = 4;
para.reg = 'exponential regularization';

para.iner = 1;


fprintf('runing TITAN_NO_EXTRAPOLATION \n');
    method = 1;
    [out{method}] = TITAN_NO_EXTRAPOLATION( train_Data, lambda, theta, para );

fprintf('runing TITAN_EXTRAPOLATION \n');
    method = 2;
    [out{method}] = TITAN_EXTRAPOLATION( train_Data, lambda, theta, para );

fprintf('runing PALM \n');
    method = 3;
    [out{method}] = PALM( train_Data, lambda, theta, para );


figure;
subplot(1, 2, 1);
hold on;

plot(out{1}.Time, log(out{1}.obj), 'r');


plot(out{2}.Time, log(out{2}.obj), 'g');

plot(out{3}.Time, log(out{3}.obj), 'b');



legend('TITAN-NO', 'TITAN-EXTRA','PALM');

xlabel('CPU time (s)');
ylabel('Objective value (log scale)');
title('movielens1m')

figure;
subplot(1, 2, 1);
hold on;

plot(out{1}.Time, out{1}.RMSE, 'r');


plot(out{2}.Time, out{2}.RMSE, 'g');

plot(out{3}.Time, out{3}.RMSE, 'b');



legend('TITAN-NO', 'TITAN-EXTRA','PALM');

xlabel('CPU time (s)');
ylabel('RMSE');
title('movielens1m')
    
