% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Q02 - simulation for Go/No Go Task
clc
clear
close all

num_iters = 20;
sigma = 1;
dt = 0.1;
time_interval = 0:dt:1;

bias = 0;
X = zeros(length(time_interval), num_iters);
choices = zeros(1, num_iters);
for iter = 1:num_iters
    [X(:, iter), choices(1, iter)] = simple_model(bias, sigma, dt, time_interval);
end
figure
histogram(X, 'Normalization', 'pdf')
xlabel('X')
ylabel('Probability Density')
title("PDF of X | B = " + num2str(bias))
figure
qqplot(reshape(X, [], 1))


bias = 1;
X = zeros(length(time_interval), num_iters);
choices = zeros(1, num_iters);
for iter = 1:num_iters
    [X(:, iter), choices(1, iter)] = simple_model(bias, sigma, dt, time_interval);
end
figure
histogram(X, 'Normalization', 'pdf')
xlabel('X')
ylabel('Probability Density')
title("PDF of X | B = " + num2str(bias))
figure
qqplot(reshape(X, [], 1))

figure
plot(time_interval, mean(X, 2), 'k', 'LineWidth', 2)
hold on
plot(time_interval, X', '--k')
legend('Avg of all Iteratoins', 'Differnet Iterations', 'Location', 'northwest')
xlabel('Time (s)')
ylabel('X')
%% 10 second trial
clc
close all

sigma = 1;
dt = 0.1;
time_interval = 0:dt:10;
biases = [-1, 0, 0.1, 1, 10];
X = zeros(length(biases), length(time_interval));

bias_index = 1;
for bias = biases
    [X(bias_index, :), ~] = simple_model(bias, sigma, dt, time_interval);
    bias_index = bias_index+1;
end

plot(time_interval, X, 'LineWidth', 2)
xlabel('Time (s)')
ylabel('X')
legend('B = -1', 'B = 0', 'B = 0.1', 'B = 1', 'B = 10', 'Location', 'northwest')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Q03 - Relatoin between Error rate and time
clc
close all

num_iters = 1000;
sigma = 1;
dt = 0.1;
bias = 0.1;
times = 0.5:1:10;
error = zeros(1, length(times));
time_index = 1;
for time = times
    time_interval = 0:dt:time;
    choices = zeros(1, num_iters);
    X = zeros(length(time_interval), num_iters);
    for iter = 1:num_iters
        [X(:, iter), choices(1, iter)] = simple_model(bias, sigma, dt, time_interval);
    end
    error(1, time_index) = 1-sum(choices == sign(bias))/length(choices);
    time_index = time_index+1;
end

plot(times, error, 'k', 'LineWidth', 2)
xlabel('Time Intervals')
ylabel('Error (percantage of wrong choices)')
ylim([0,0.6])
