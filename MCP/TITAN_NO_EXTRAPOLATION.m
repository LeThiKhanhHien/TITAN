%% TITAN for solving matrix completion problem (MCP) without extrapolation
% Reference: LTK Hien, DN Phan, N Gillis. "An Inertial Block Majorization Minimization Framework for 
% Nonsmooth Nonconvex  Optimization".
%
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


function [output ] = TITAN_NO_EXTRAPOLATION( D, lam, theta, para )
% D: sparse observed matrix

output.method = 'TITAN_NO_EXTRAPOLATION';

objstep = 0.01;

maxIter = para.maxIter;
tol = para.tol*objstep;

[row, col, data] = find(D);

[m, n] = size(D);

U0 = para.U0;
U1 = U0;

V0 = para.V0';
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

ga = theta;
fun_num = para.fun_num;
part0 = data - part0';
obj(1) = obj_func(part0, U0, V0, lam, fun_num, ga);

L = 1;

Lls(1,1) = L;
Lls(1,2) = L;

 % testing performance
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


for i = 1:maxIter
    tt = cputime;

    L1 = norm(V1*V1','fro');
    L1 = max(L1,1e-4);
        
    for iner_iter = 1:para.iner

        setSval(spa,part0,length(part0));

        grad_U = -spa*V1';

        tU1 = U1 - grad_U/(L*L1);

        wu = lam*ga*exp(-ga*abs(U1));

        U1 = max(0,abs(tU1) - wu/(L*L1)).*sign(tU1);

        part0 = sparse_inp(U1', V1, row, col);

        part0 = data - part0';
    end
    
    %%%update V
    L2 = norm(U1'*U1,'fro');
    L2 = max(L2,1e-4);
    
    for iner_iter = 1:para.iner
    
        setSval(spa,part0,length(part0));

        grad_V = -U1'*spa;

        tV1 = V1 - grad_V/(L*L2);

        wv = lam*ga*exp(-ga*abs(V1));

        V1 = max(0,abs(tV1) - wv/(L*L2)).*sign(tV1);

        part0 = sparse_inp(U1', V1, row, col);
        
        part0 = data - part0';
    end
    
    x_obj = obj_func(part0, U1, V1, lam, fun_num, ga);
    
    Lls(i+1,1) = L1;
    Lls(i+1,2) = L2;

    if(i > 1)
        delta = (obj(i)- x_obj)/x_obj;
    else
        delta = inf;
    end
    
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
            trainRMSE(i+1) = sqrt(sum(part0.^2)/length(data));
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


