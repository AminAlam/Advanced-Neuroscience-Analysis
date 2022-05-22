function winner_no = race_trial_with_time_limit(threshold, bias, sigma, X_0, dt, time_limit)
    X = X_0;
    t = 0;
    while X < threshold & t < time_limit
        dW = normrnd(0,sigma,2,1);
        dX = bias*dt+sigma*dW;
        X = X + dX;
        t = t+dt;
    end
    
    [winner_no, ~] = find(X > threshold);
    
    if length(winner_no) > 1
        winner_no = 0;
    elseif t > time_limit
        winner_no = 3;
    end
end