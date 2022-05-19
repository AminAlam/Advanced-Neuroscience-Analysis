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
random_gen_data = normrnd(bias*time_interval(end), sigma*time_interval(end), size(X,1), size(X,2));
figure
histogram(X, 'Normalization', 'pdf')
hold on 
histogram(random_gen_data, 'Normalization', 'pdf')
xlabel('X')
ylabel('Probability Density')
title("PDF of X | B = " + num2str(bias))
legend('X', 'N(0, \sigma)')
figure
qqplot(reshape(X, [], 1))

bias = 1;
X = zeros(length(time_interval), num_iters);
choices = zeros(1, num_iters);
for iter = 1:num_iters
    [X(:, iter), choices(1, iter)] = simple_model(bias, sigma, dt, time_interval);
end
random_gen_data = normrnd(bias*time_interval(end), sigma*time_interval(end), size(X,1), size(X,2));
figure
histogram(X, 'Normalization', 'pdf')
hold on 
histogram(random_gen_data, 'Normalization', 'pdf')
xlabel('X')
ylabel('Probability Density')
title("PDF of X | B = " + num2str(bias))
legend('X', 'N(1, \sigma)')
figure
qqplot(reshape(X, [], 1))

figure
plot(time_interval, mean(X, 2), 'k', 'LineWidth', 3)
hold on
plot(time_interval, X', '--k')
legend('Avg of all Iteratoins', 'Different Iterations', 'Location', 'northwest')
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
times = 0.5:1:50;
error = zeros(1, length(times));
error_theory = zeros(1, length(times));
time_index = 1;

for time = times
    time_interval = 0:dt:time;
    choices = zeros(1, num_iters);
    X = zeros(length(time_interval), num_iters);
    for iter = 1:num_iters
        [X(:, iter), choices(1, iter)] = simple_model(bias, sigma, dt, time_interval);
    end
    error(1, time_index) = 1-sum(choices == sign(bias))/length(choices);
    
    mean_theory = bias*time;
    sigma_theory = sqrt(dt*time);
    error_prob = 1 - min(8*sigma_theory, mean_theory+4*sigma_theory)/(8*sigma_theory);
    error_theory(1, time_index) = error_prob;
    
    time_index = time_index+1;
end

plot(times, error, 'k', 'LineWidth', 2)
hold on
plot(times, error_theory, 'r', 'LineWidth', 2)
xlabel('Time Intervals')
ylabel('Error (percantage of wrong choices)')
legend('Simulation', 'Theory')
ylim([0,0.6])
%% plots for report - theory part proof






