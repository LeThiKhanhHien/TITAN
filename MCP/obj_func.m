function[ObjVal] = obj_func(part, U, V, lam, fun_num, ga)
    
    if(fun_num == 4)
        ObjVal = (1/2)*sum(part.^2) + lam*(sum(sum(1 - exp(-ga*abs(U)))) + sum(sum(1 - exp(-ga*abs(V)))));
    end

end