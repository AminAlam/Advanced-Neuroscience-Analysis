function [t, choice] = two_choice_trial(thresholds, bias, sigma, X_0, dt)
    X = X_0;
    t = 0;
    while X > thresholds(1) && X < thresholds(2)
        dW = normrnd(0,sigma,1,1);
        dX = bias*dt+sigma*dW;
        X = X + dX;
        t = t+dt;
    end
    choice = sign(X);
end