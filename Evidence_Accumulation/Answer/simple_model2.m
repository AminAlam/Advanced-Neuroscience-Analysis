function p = simple_model2(bias, sigma, X_0, time_limit)
        
        mu = bias*time_limit;
        sigma = sqrt(sigma*time_limit);
        p = normcdf(X_0,mu,sigma);

        p = 1-p;



end