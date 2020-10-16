%% PALM for solving sparsity NMF
% Reference: J. Bolte, S. Sabach, and M. Teboulle. Proximal alternating linearized minimization for 
% non-convex and nonsmooth problems. Mathematical Programming, 146(1):459{494, Aug 2014
% 
% Written by LTK Hien, Umons, Belgium.
% Latest update October 2020
% 
% Input 
%   X: input data matrix
%   r: rank
%   sparsity: percentage of non-zero elements in each column of U
%   options: a structure including  
%               'display' (1 if fitting error is showed to the screen during the run of PALM)
%               'init.W', 'init.H' (initial points of U and V)
%               'maxiter' (maximum number of iterations)
%               'timemax' (maximum of running time)
% Output
%  (W,H): solution
%      e: sequence of fitting errors
%      t: corresponding running time
function [W,H,e,t] = PALM(X,r,sparsity,options) 

cputime0 = tic; 
[m,n]=size(X);
%% Parameters of NMF algorithm
if nargin < 3
    options = [];
end
if ~isfield(options,'display')
    options.display = 1; 
end
if ~isfield(options,'init')
    W = rand(m,r); 
    H = rand(r,n); 
else
    W = options.init.W; 
    H = options.init.H; 
end
if ~isfield(options,'maxiter')
    options.maxiter = 200; 
end
if ~isfield(options,'timemax')
    options.timemax = Inf; 
end

%% Main loop
sparsity_number=round(sparsity*m);

nX = norm(X,'fro'); 
i = 1; 

% scale the innitial point 
HHt = H*H'; 
XHt = X*H'; 
scaling = sum(sum(XHt.*W))/sum(sum( (W'*W).*(HHt) )); 
W = W*scaling; 

time1=tic;
e(1)= nX^2 - 2*sum(sum( (W*H).*X ) ) + sum(sum( (W'*W).*(H*H') ) );
e(1)= sqrt(max(0,e(1)))/nX; % e is to save relative error
time_err=toc(time1);
t(1) = toc(cputime0)-time_err;
while i <= options.maxiter && t(i) < options.timemax  
   
    %% Update Wn
   
    HHt = H*H'; 
    XHt = X*H'; 
    Lw = norm(HHt); 
    gradW = W*HHt - XHt; 
    W= max( 0 , W - gradW/Lw  );
    W=proj_l0_col(W,r,sparsity_number);
   
    %% Update Hn
  
    WtW = W'*W; 
    WntX = W'*X; 
    Lh= norm(WtW); 
    gradH = WtW*H - WntX;  
    H = max( 0 , H - gradH/Lh  ); 
   
    time1=tic;
    e(i+1)= nX^2 - 2*sum(sum( (W*H).*X ) ) + sum(sum( (W'*W).*(H*H') ) );
    e(i+1)=sqrt(max(0,e(i+1)))/nX;
    
    time_err=time_err + toc(time1);
    t(i+1) = toc(cputime0)-time_err; 
    
    %% Display iteration number and error in percent
    if mod(i,10)==0 && options.display==1
        fprintf('PALM: iteration %4d fitting error: %1.2e\n',i,e(i));     
    end
    i = i + 1; 
end

end
