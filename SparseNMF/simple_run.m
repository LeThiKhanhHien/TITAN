% simple run 
clear all; close all; clc; 
options.maxiter =inf;
options.timemax =8;

m=200; 
n=200; 
r = 20;  %rank 
sparsity=0.3; 

% generate M
sparsity_number=round(sparsity*m);
U=rand(m,r); 
U=proj_l0_col(U,r,sparsity_number);
V=rand(r,n);
M = U*V; 

% initialization 
 W=rand(m,r);
 H=rand(r,n);
 options.init.W = W;
 options.init.H = H;
 
 fprintf('Running TITAN kappa = 1.0001  \n'); 
 [~,~,e1,t1] =  TITAN_Nesterov(M,r,sparsity,options,1.0001);
 
 fprintf('Running PALM, problem  \n'); 
 [~,~,e2,t2] =PALM(M,r,sparsity,options);
 
 fprintf('Running TITAN kappa  \n'); 
 [W,H,e3,t3] =  TITAN_Nesterov(M,r,sparsity,options,1.5);
 
 emin=min([min(e1),min(e2),min(e3)]);
 set(0, 'DefaultAxesFontSize', 18);
 set(0, 'DefaultLineLineWidth', 2);
 figure;
 semilogy(t1,e1-emin,'y-','linewidth',3);hold on; 
 semilogy(t2,e2-emin,'r-','linewidth',3);hold on; 
 semilogy(t3,e3-emin,'b-','linewidth',3);hold on; 
 ylabel('||M-UV||/||M||-e_{min}');
 xlabel('times')
 legend('TITAN - kappa=1.0001','PALM','TITAN - kappa = 1.5');  
