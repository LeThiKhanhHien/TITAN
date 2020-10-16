function[U] = make_update_palm(U, tgrad_U,L,lam,ga,fun_num)
    
    if(fun_num==4)
        
        maxiter = 30;
        for iter = 1:maxiter
            w = lam*ga*exp(-ga*abs(U));
            grad = tgrad_U;
            U = max(0,abs(grad) - w/L).*sign(grad); 
            
        end
        
       
    end

end


