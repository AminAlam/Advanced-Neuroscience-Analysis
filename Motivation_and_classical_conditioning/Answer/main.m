%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RW rule 
% Extinction
clc
clear
close all

figure
num_trials = 200;
espsilon = 5e-2;
u = ones(1, num_trials);
r = ones(1, num_trials);
r(end/2:end) = 0;
w0 = 0;
w = deltaRule(w0,num_trials,espsilon,u,r);
plot(1:num_trials, w,'k', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('\omega')
title('Extinction')

hold on

box = [0 0 100 100];
boxy = [1 0 0 1];
patch(box,boxy,'r','FaceAlpha',0.1)
xl = xline(0, 'r', 'Pre Training');
xl.LabelVerticalAlignment = 'middle';

box = [100 100 200 200];
boxy = [1 0 0 1];
patch(box,boxy,'b','FaceAlpha',0.1)
xl = xline(100, 'b', 'Training');
xl.LabelVerticalAlignment = 'middle';
%% partial
clc
close all
figure
num_trials = 200;
espsilon = 5e-2;
alpha = 0.1;
u = ones(1, num_trials);
r = rand(1, num_trials) < alpha;
w0 = 0;
w = deltaRule(w0,num_trials,espsilon,u,r);
scatter(1:num_trials, w, 'k', 'LineWidth', 2)

hold on

alpha = 0.4;
u = ones(1, num_trials);
r = rand(1, num_trials) < alpha;
w0 = 0;
w = deltaRule(w0,num_trials,espsilon,u,r);
scatter(1:num_trials, w, 'b', 'LineWidth', 2)


alpha = 0.7;
u = ones(1, num_trials);
r = rand(1, num_trials) < alpha;
w0 = 0;
w = deltaRule(w0,num_trials,espsilon,u,r);
scatter(1:num_trials, w,'m', 'LineWidth', 2)

yline(0.1, '--k', 'LineWidth', 1);
yline(0.4, '--b', 'LineWidth', 1);
yline(0.7, '--m', 'LineWidth', 1);

xlabel('Trial Number')
ylabel('\omega')
title('Extinction')
legend('\alpha = 0.1', '\alpha = 0.4', '\alpha = 0.7')
%% blocking
clc
close all
figure
num_trials = 200;
espsilon = 5e-2;
u = ones(2, num_trials);
r = ones(1, num_trials);
u(2, 1:end/2) = 0;
w0 = [0;0];
w = deltaRule(w0,num_trials,espsilon,u,r);
plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
plot(1:num_trials, w(2,:), 'm', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('\omega')
title('Blocking')

box = [0 0 100 100];
boxy = [1 0 0 1];
patch(box,boxy,'r','FaceAlpha',0.1)
xl = xline(0, 'r', 'Pre Training');
xl.LabelVerticalAlignment = 'middle';

box = [100 100 200 200];
boxy = [1 0 0 1];
patch(box,boxy,'b','FaceAlpha',0.1)
xl = xline(100, 'b', 'Training');
xl.LabelVerticalAlignment = 'middle';

legend('\omega_1', '\omega_2')
%% inhibitory
clc
close all
figure
num_trials = 200;
espsilon = 5e-2;
alpha = 0.1;
u = ones(2, num_trials);
u(2, :) = rand(1, num_trials) < 0.5;
r = zeros(1, num_trials);
for i = 1:size(u,2)
    if u(1, i) + u(2, i) ~= 2 && u(1, i) == 1
       r(1, i) = 1;
    end
end

w0 = [0;0];
w = deltaRule(w0,num_trials,espsilon,u,r);
plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
plot(1:num_trials, w(2,:), 'b', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('w')
title('Inhibitory')
legend('\omega_1', '\omega_2')
%% overshadow
clc
close all
figure
num_trials = 200;
espsilon = 5e-2;
alpha1 = 0.5;
alpha2 = 0.5;
u = zeros(2, num_trials);
u(1, :) = rand(1, num_trials) < alpha1;
u(2, :) = rand(1, num_trials) < alpha2;
r = zeros(1, num_trials);
for i = 1:size(u,2)
    if u(1, i) + u(2, i) == 2
       r(1, i) = 1;
    end
end

w0 = [0;0];
w = deltaRule(w0,num_trials,espsilon,u,r);
plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
plot(1:num_trials, w(2,:), 'b', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('\omega')
title('Overshadow | \alpha_1 = 0.5 , \alpha_2 = 0.5')
legend('\omega_1', '\omega_2')
ylim([0, 1])

figure
num_trials = 200;
espsilon = 5e-2;
alpha1 = 0.9;
alpha2 = 0.3;
u(1, :) = rand(1, num_trials) < alpha1;
u(2, :) = rand(1, num_trials) < alpha2;
r = zeros(1, num_trials);
for i = 1:size(u,2)
    if u(1, i) + u(2, i) == 2
       r(1, i) = 1;
    end
end

w0 = [0;0];
w = deltaRule(w0,num_trials,espsilon,u,r);
plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
plot(1:num_trials, w(2,:), 'b', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('\omega')
title('Overshadow | \alpha_1 = 0.9 , \alpha_2 = 0.4')
legend('\omega_1', '\omega_2')
ylim([0, 1])

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kalman Filter
clc
clear
close all

% Figure 1.b of the paper
num_trials = 20;
tau = 0.1;
v = [normrnd(0, tau , [1, num_trials]); normrnd(0, 0.1, [1, num_trials])];
w0 = [1;1];
w = zeros(size(w0,1),num_trials);
w(:,1) = w0;

for trial_no = 2:num_trials
    w(:, trial_no)  = w(:, trial_no-1) - v(:, trial_no-1);
end
t_5 = 5;
hist_plot = histogram(w(:, 1:t_5), 'Normalization', 'pdf');
y_value_5 = hist_plot.Values;
t_18 = 18;
hist_plot = histogram(w(:, 1:t_18), 'Normalization', 'pdf');
y_value_18 = hist_plot.Values;

plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
plot(1:num_trials, w(2,:), '--k', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('\omega')
xlim([1, num_trials])
ylim([0, 2])
title('Drift')
plot(y_value_5/2+5, linspace(0, 2, length(y_value_5)), 'k')
xline(5, '--k');
plot(y_value_18/2+18, linspace(0, 2, length(y_value_18)), 'k')
xline(18, '--k');
legend('\omega_1', '\omega_2')

figure
num_trials = 20;
u = ones(2, num_trials);
r = ones(1, num_trials);
u(2, 1:end/2) = 0;
w0 = [0;0];
tau = 0.5;
cov0 = eye(2)*0.6;
[w, cov_mat] = kalamFilter(u, r, cov0, w0, tau, num_trials);

plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
plot(1:num_trials, w(2,:), '--k', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('\omega')
legend('\omega_1', '\omega_2')
title('Blocking')

figure

plot(1:num_trials, squeeze(cov_mat(1,1,:)), 'k', 'LineWidth', 2)
hold on
plot(floor(num_trials/2)+1:num_trials, squeeze(cov_mat(2,2,floor(num_trials/2)+1:end)), '--k', 'LineWidth', 2)
xline(floor(num_trials/2)+1, '--k');
ylim([0, 1]);
xlim([1, num_trials])
xlabel('Trial Number')
ylabel('\sigma^2')
legend('\sigma_1^2', '\sigma_2^2')
title('Blocking')


