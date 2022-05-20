function winner_no = race_trial(threshold, bias, sigma, X_0, dt)
    X = X_0;
    t = 0;
    while X < threshold 
        dW = normrnd(0,sigma,2,1);
        dX = bias*dt+sigma*dW;
        X = X + dX;
        t = t+dt;
    end
    [winner_no, ~] = find(X > threshold);
    if length(winner_no) > 1
        winner_no = 0;
    end
end