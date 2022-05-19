function [X, choice] = simple_model(bias, sigma, dt, time_interval)
    dW = normrnd(0,sqrt(dt),1,length(time_interval));
    X = zeros(1, length(time_interval));
    for i = 1:1:length(time_interval)-1
        dX = bias*dt+sigma*dW(i);
        X(i+1) = X(i)+dX;
    end
    choice = sign(X(end));
end