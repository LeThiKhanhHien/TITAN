%% PALM for solving matrix completion problem (MCP)
% Reference: J. Bolte, S. Sabach, and M. Teboulle. Proximal alternating linearized minimization for 
% non-convex and nonsmooth problems. Mathematical Programming, 146(1):459{494, Aug 2014

% Written by Duy Nhat Phan.
% Latest update September 2020
%
% Input 
%   D: sparse observed matrix (training set)
%   para.test....: test set information
%   para.maxR: rank
%   regularization: exponential regularization with parameters lambda and
%   theta
%   para: 
%        para.U0, para.V0: initial point
%        para.maxIter: maximum number of iterations
%        para.maxtime: maximum of running time
%       
% Output
%  (output.U, output.V): solution
%  At each iteration:
%  output.RMSE: root mean square error on test set
%  output.trainRMSE: root mean square error on training set
%  output.Time: corresponding running time
%  output.obj: objective function value

function [output ] = PALM( D, lam, theta, para )
% D: sparse observed matrix

output.method = 'PALM';

objstep = 1;

maxIter = para.maxIter;
tol = para.tol*objstep;

[row, col, data] = find(D);

[m, n] = size(D);

U0 = para.U0;
U1 = U0;

[~, ~, V0] = svd(U0'*D, 'econ');
V0 = V0';
V1 = V0;

spa = sparse(row, col, data, m, n); % data input == D

clear D;

obj = zeros(maxIter+1, 1);
RMSE = zeros(maxIter+1, 1);
trainRMSE = zeros(maxIter+1, 1);
Time = zeros(maxIter+1, 1);
Lls = zeros(maxIter+1, 2);

nnzUV = zeros(maxIter+1, 2);

part0 = partXY(U0', V0, row, col, length(data));
part0 = data - part0';

ga = theta;
fun_num = para.fun_num;

obj(1) = obj_func(part0, U0, V0, lam, fun_num, ga);


L = 1;
if(isfield(para, 'test'))
    tempS = eye(size(U1,2), size(V1',2));
    if(para.test.m ~= m)
        RMSE(1) = MatCompRMSE(V1', U1, tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum(part0.^2)/length(data));
    else
        RMSE(1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum(part0.^2)/length(data));
    end
    fprintf('method: %s data: %s  RMSE %.2d \n', output.method, para.data, RMSE(1));
end

Lls(1,1) = L;
Lls(1,2) = L;

for i = 1:maxIter
    tt = cputime;

    setSval(spa,part0,length(part0));

    grad_U = -spa*V1';
    
    L1 = norm(V1*V1','fro');
    L1 = max(L1,1e-4);
    
    tU1 = U1 - grad_U/(L*L1);
    
    [U1] = make_update_palm(U1,tU1,L*L1,lam,ga,fun_num);
    
    part0 = sparse_inp(U1', V1, row, col);
    
    part0 = data - part0';
    
    setSval(spa,part0,length(part0));
    
    grad_V = -U1'*spa;
    
    L2 = norm(U1'*U1,'fro');
    L2 = max(L2,1e-4);
    
    tV1 = V1 - grad_V/(L*L2);
    
    [V1] = make_update_palm(V1,tV1,L*L2,lam,ga,fun_num);

    part0 = sparse_inp(U1', V1, row, col);
    
    part0 = data - part0';

    x_obj = obj_func(part0, U1, V1, lam, fun_num, ga);
        

    if(i > 1)
        delta = (obj(i)- x_obj)/x_obj;
    else
        delta = inf;
    end
    
    Lls(i+1,1) = L1;
    Lls(i+1,2) = L2;
    
    Time(i+1) = cputime - tt;
    obj(i+1) = x_obj;
    
   nnzUV(i+1,1) = nnz(U1)/(size(U1,1)*size(U1,2));
    nnzUV(i+1,2) = nnz(V1)/(size(V1,1)*size(V1,2));
    
    
    
    % testing performance
    if(isfield(para, 'test'))
        tempS = eye(size(U1,2), size(V1',2));
        if(para.test.m ~= m)
            RMSE(i+1) = MatCompRMSE(V1', U1, tempS, para.test.row, para.test.col, para.test.data);
            trainRMSE(i+1) = sqrt(sum(part0.^2)/length(data));
        else
            RMSE(i+1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
            trainRMSE(1) = sqrt(sum(part0.^2)/length(data));
        end
        fprintf('method: %s data: %s  RMSE %.2d \n', output.method, para.data, RMSE(i));
    end
    
    if(i > 1 && abs(delta) < tol)
        break;
    end
    
    if(sum(Time) > para.maxtime)
        break;
    end
end

output.obj = obj(1:(i+1));
output.Rank = para.maxR;
output.RMSE = RMSE(1:(i+1));
output.trainRMSE = trainRMSE(1:(i+1));

Time = cumsum(Time);
output.Time = Time(1:(i+1));
output.U = U1;
output.V = V1;
output.data = para.data;
output.L = Lls(1:(i+1),:);
output.nnzUV = nnzUV(1:(i+1),:);
output.lambda = lam;
output.theta = ga;
output.reg = para.reg;

end


