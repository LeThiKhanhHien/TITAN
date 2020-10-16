%% proj_l0_col(W,r,s)
% Keep the s largest values in each column of W and set the remaining values to zero.
% W has r columns.
% 
% written by LTK Hien, Umons, Belgium.
% Latest update September 2020.
%
function  W=proj_l0_col(W,r,s)
    for i=1:r
        [b,I]=sort(W(:,i),'descend');
        b(s+1:end)=0;
        wi(I)=b;
        W(:,i)=wi;
    end

end