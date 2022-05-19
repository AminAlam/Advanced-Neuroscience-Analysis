function [X, choice] = simple_model(bias, sigma, dt, time_interval)
    dW = normrnd(0,sqrt(dt),1,floor(time_interval/dt));
    X = zeros(1, floor(time_interval/dt));
    for i = 1:1:floor(time_interval/dt)-1
        dX = bias*dt+sigma*dW(i);
        X(i+1) = X(i)+dX;
    end
    choice = sign(X(end));
end