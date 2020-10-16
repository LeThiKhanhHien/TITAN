%% TITAN for solving sparsity NMF using Nesterov type extrapolation
% Reference: LTK Hien, DN Phan, N Gillis. "An Inertial Block Majorization Minimization Framework for 
% Nonsmooth Nonconvex  Optimization".
%
% Written by LTK Hien, Umons, Belgium.
% Latest update September 2020
%
% Input 
%   X: input data matrix
%   r: rank
%   sparsity: percentage of non-zero elements in each column of U
%   options: a structure including  
%            'display' (1 if fitting error is showed on the screen during the run of PALM)
%            'init.W', 'init.H' (initial points of U and V)
%             'maxiter' (maximum number of iterations)
%             'timemax' (maximum of running time)
%             'delta' (the parameter to stop the inner loop of repeating
%                     update W or H. Criteria: if ||W-W_old|| <= delta ||W-W_start|| then stop. 
%                     By default delta = 0.01) 
%             'alphaparam' (the parameter to control the max iterate of
%                      the inner loop. By defaut alphaparam=0.3. 
%   kappa: the parameter in choosing extrapolation parameter, 
%           by default kappa = 1.0001
% Output
%  (W,H): solution
%      e: sequence of relative fitting errors
%      t: corresponding running time

function [W,H,e,t] = TITAN_Nesterov(X,r,sparsity,options,kappa) 

cputime0 = tic; 
[m,n]=size(X);
kappa_1=kappa-1;
%% Parameters of NMF algorithm
if nargin < 4
    options = [];
end
if nargin < 5
    kappa=1.0001;
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
if ~isfield(options,'delta')
    options.delta = 0.01; 
end
if ~isfield(options,'alphaparam')
    options.alphaparam = 0.3; 
end

delta=options.delta; % to check when to stop the inner loop 
alphaparam=options.alphaparam; % to control the number of inner loop.  
%% Main loop
sparsity_number=round(sparsity*m);
nX = norm(X,'fro'); 
i = 1; 
paramt(i) = 1; 
% scale the innitial point 
HHt = H*H'; 
XHt = X*H'; 
scaling = sum(sum(XHt.*W))/sum(sum( (W'*W).*(HHt) )); 
W = W*scaling; 
Wold=W;
Hold=H;

time1=tic;
e(1)= nX^2 - 2*sum(sum( (W*H).*X ) ) + sum(sum( (W'*W).*(H*H') ) );
e(1)= sqrt(max(0,e(1)))/nX; % e is to save relative error
time_err=toc(time1);
t(1) = toc(cputime0)-time_err;

KX = sum( X(:) > 0 );


inneriterH= floor( 1 + alphaparam*(KX+m*r)/(n*r+n) );
inneriterW= floor( 1 + alphaparam*(KX+n*r)/(m*r+m) );
deltaw = 0.5; % approximate sqrt(C*nu*(1-nu))
while i <= options.maxiter && t(i) < options.timemax  
   
    %% Update Wn
    paramt(i+1) = 0.5 * ( 1+sqrt( 1 + 4*paramt(i)^2 ) ); 
    what(i) = (paramt(i)-1)/paramt(i+1); 

    HHt = H*H'; 
    XHt = X*H'; 
    Lw(i) = norm(HHt); 
    if i == 1
        ww(i) = what(i); 
    else
        ww(i) = min( what(i), deltaw*kappa_1/kappa*sqrt( Lw(i-1)/Lw(i) ) ); 
    end
      j=1;eps0 = 0; eps = 1; 

    while j<=inneriterW &&  eps >= delta*eps0
        WWold=W-Wold;
        eps= norm(WWold,'fro');
 
        Wex=W +ww(i)*WWold;
        gradW = Wex*HHt - XHt; 
        
        Wold=W;
        W= max( 0 , Wex - gradW/Lw(i)/kappa  );
        W=proj_l0_col(W,r,sparsity_number);
         if j==1 
             eps0=eps;
         end
          j=j+1;
     end
    
    %% Update Hn
  
    WtW = W'*W; 
    WntX = W'*X; 
    Lh(i)= norm(WtW); 
    if i == 1
        wh(i) = what(i); 
    else
        wh(i) = min( what(i), 0.9999*sqrt( Lh(i-1)/Lh(i) ) ); 
    end

     j=1; eps0 = 0; eps = 1; 
     while j<=inneriterH &&  eps >= delta*eps0
        HHold=(H-Hold);
        eps=norm(HHold,'fro');
        Hex= H + wh(i)*HHold;
        
        gradH = WtW*Hex - WntX;  
        Hold=H;
        H = max( 0 , Hex - gradH/Lh(i)  ); 
        if j==1
           eps0=eps;
        end
        j=j+1;
    end
    time1=tic;
    e(i+1)= nX^2 - 2*sum(sum( (W*H).*X ) ) + sum(sum( (W'*W).*(H*H') ) );
    e(i+1)=sqrt(max(0,e(i+1)))/nX;
    time_err=time_err + toc(time1);
    t(i+1) = toc(cputime0)-time_err; 
    
    %% Display iteration number and error in percent
      if mod(i,100)==0 && options.display == 1
        fprintf('Titan_nesterov: iteration %4d fitting error: %1.2e\n',i,e(i));     
      end
    i = i + 1; 
end

end
