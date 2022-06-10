function basis_function = return_basis_functions(A,S_var,max_coeff)
    [L M]=size(A);
    sz=sqrt(L);
    buf=1;
    if floor(sqrt(M))^2 ~= M
      m=sqrt(M/2);
      n=M/m;
    else
      m=sqrt(M);
      n=m;
    end

    k=1;

    for i=1:m
      for j=1:n
        clim=max(abs(A(:,k)));
    
        if k == max_coeff
            basis_function = reshape(A(:,k),sz,sz)/clim;
        	return
        end
        k=k+1;
      end
    end
    
end