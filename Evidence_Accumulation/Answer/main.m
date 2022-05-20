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
clc
close all

save_figures = 1;

x = -5:.1:5;
y = normpdf(x,0,1);
plot(x,y, 'k', 'LineWidth', 2)

xl = xline(0, '--', '\mu');
xl.LabelVerticalAlignment = 'middle';
xl.LabelHorizontalAlignment = 'center';
xl.FontSize = 20;

xl = xline(-4, '--', '\mu + -4\sigma');
xl.LabelVerticalAlignment = 'middle';
xl.LabelHorizontalAlignment = 'center';
xl.FontSize = 20;

xl = xline(4, '--', '\mu + 4\sigma');
xl.LabelVerticalAlignment = 'middle';
xl.LabelHorizontalAlignment = 'center';
xl.FontSize = 20;

xlim([-5 5])
ylim([0, 0.5])

set(gca,'xtick',[])
set(gca,'ytick',[])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/normal_proof','-dpng','-r0')
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Q04 - Mean and variance of X in different times
clc
close all

num_iters = 1000;
sigma = 1;
dt = 0.1;
time_interval = 0:dt:10;
bias = 0.1;

X = zeros(length(time_interval), num_iters);
for iter = 1:num_iters
    [X(:, iter), ~] = simple_model(bias, sigma, dt, time_interval);
end

mean_X = mean(X, 2);
var_X = var(X, 1, 2);

subplot(3,1,1)
plot(time_interval, X, 'k')
xlabel('Time (s)')
title('X(t) during different trials')

subplot(3,1,2)
plot(time_interval, mean_X, 'k', 'LineWidth', 2)
hold on
plot(time_interval, bias*time_interval, '--r', 'LineWidth', 2)
xlabel('Time (s)')
title('Expected Value of X(t)')
legend('Simulation', 'Theory', 'Location', 'northwest')

subplot(3,1,3)
plot(time_interval, var_X, 'k', 'LineWidth', 2)
hold on
plot(time_interval, sigma*time_interval, '--r', 'LineWidth', 2)
xlabel('Time (s)')
title('Variance of X(t)')
legend('Simulation', 'Theory', 'Location', 'northwest')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Q05 - Calculating the probability of right choice 
%                                       using the CDF of normal distribution 
clc
close all

bias = 0;
sigma = 1;
Xs_0 = -10:1:10;
time_limits = 1:1:100;

probs = zeros(length(time_limits), length(Xs_0));

X_0_index = 1;
for X_0 = Xs_0
    time_limit_index = 1;
    for time_limit = time_limits
        p = simple_model2(bias, sigma, X_0, time_limit);
        probs(time_limit_index, X_0_index) = p;
        time_limit_index = time_limit_index+1;
    end
    X_0_index = X_0_index+1;
end

figure
hold on
map = bone(length(time_limits));
for time_limit_index = 1:length(time_limits)
    plot(Xs_0, probs(time_limit_index, :), 'color', map(time_limit_index, :), 'LineWidth', 1)
end
xlabel('Start Point')
ylabel('Probability')
ylim([0, 1])    
colormap(map)
c = colorbar('Ticks', [time_limits(1), time_limits(end/2), time_limits(end)], 'TickLabels', ...
        {num2str(time_limits(1)), num2str(time_limits(end/2)), num2str(time_limits(end))});
c.Label.String = 'Time Limit (s)';
caxis([time_limits(1), time_limits(end)])

figure
imagesc(Xs_0, time_limits, probs)
set(gca,'YDir','normal')
xlabel('Start Point')
ylabel('Time Limit (s)')
colormap bone
c = colorbar;
c.Label.String = 'Probability';


% 3D plot for showing effect of bias
biases = -0.2:0.1:0.2;
probs = zeros(length(biases), length(time_limits), length(Xs_0));
bias_index = 1;

for bias = biases
    X_0_index = 1;
    for X_0 = Xs_0
        time_limit_index = 1;
        for time_limit = time_limits
            p = simple_model2(bias, sigma, X_0, time_limit);
            probs(bias_index, time_limit_index, X_0_index) = p;
            time_limit_index = time_limit_index+1;
        end
        X_0_index = X_0_index+1;
    end
    bias_index = bias_index+1;
end

figure

for bias_index = 1:length(biases)
    for time_limit_index = 1:length(time_limits)
        plot3(zeros(size(Xs_0))+biases(bias_index), Xs_0, squeeze(probs(bias_index, time_limit_index, :)), 'color', map(time_limit_index, :), 'LineWidth', 1)
        hold on
    end
end
xlabel('Bias')
ylabel('Start Point')
zlabel('Probability')
colormap bone
c = colorbar;
c.Label.String = 'Time Limit (s)';

for bias_index = 1:length(biases)
    figure
    for time_limit_index = 1:length(time_limits)
        plot(Xs_0, squeeze(probs(bias_index, time_limit_index, :)), 'color', map(time_limit_index, :), 'LineWidth', 1)
        hold on
    end
    xlabel('Start Point')
    ylabel('Probability')
    colormap bone
    c = colorbar;
    c.Label.String = 'Time Limit (s)';
    title("Bias = "+num2str(biases(bias_index)))
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Q07 - Simulating situation of free response
clc
close all
num_iters = 10000;
thresholds = [-10, 10];
bias = 0.1;
sigma = 1;
X_0 = 0;
dt = 0.01;

ts = zeros(1, num_iters);
choices = zeros(1, num_iters);
for iter_no = 1:num_iters
    [t, choice] = two_choice_trial(thresholds, bias, sigma, X_0, dt);
    ts(1, iter_no) = t; 
    choices(1, iter_no) = choice; 
end

figure
hist_all = histogram(ts, 'Normalization', 'pdf');
hist_all_bins = (hist_all.BinEdges(2:end)+hist_all.BinEdges(1:end-1))/2;
hist_all_values = hist_all.Values;

hist_correct = histogram(ts(choices==1), 'Normalization', 'pdf');
hist_correct_bins = (hist_correct.BinEdges(2:end)+hist_correct.BinEdges(1:end-1))/2;
hist_correct_values = hist_correct.Values;

hist_wrong = histogram(ts(choices==-1), 'Normalization', 'pdf');
hist_wrong_bins = (hist_wrong.BinEdges(2:end)+hist_wrong.BinEdges(1:end-1))/2;
hist_wrong_values = hist_wrong.Values;

plot(hist_all_bins/dt, hist_all_values*dt, 'k', 'LineWidth', 2)
hold on
pd = makedist('InverseGaussian', 'mu', thresholds(2)/bias, 'lambda', (thresholds(2)/sigma)^2);
pdf_values = pdf(pd, hist_all_bins/dt);
plot(hist_all_bins/dt, pdf_values, 'R', 'LineWidth', 2)
xlabel('Reaction Time')
ylabel('Probability Density')
legend('Simulation', 'Theory')

figure
hold on
plot(hist_correct_bins/dt, hist_correct_values*dt, 'b', 'LineWidth', 2)
plot(hist_wrong_bins/dt, hist_wrong_values*dt, 'r', 'LineWidth', 2)
xlabel('Reaction Time')
ylabel('Probability Density')
legend('Correct choices', 'Wrong choices')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Q08 - Simulating situation of free response




