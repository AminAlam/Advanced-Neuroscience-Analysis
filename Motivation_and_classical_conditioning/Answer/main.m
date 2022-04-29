%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Q01 
%% Extinction
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
ylabel('w')
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
ylabel('w')
title('Extinction')
legend('\alpha = 0.1', '\alpha = 0.4', '\alpha = 0.7')
%% blocking
clc
close all
figure
num_trials = 200;
espsilon = 5e-2;
alpha = 0.1;
u = ones(2, num_trials);
r = ones(1, num_trials);
u(2, 1:end/2) = 0;
w0 = [0;0];
w = deltaRule(w0,num_trials,espsilon,u,r);
plot(1:num_trials, w(1,:))
hold on
plot(1:num_trials, w(2,:))
%% Functions

function w = deltaRule(w0,num_trials,espsilon,u,r)
    w = zeros(size(w0,1),num_trials);
    w(:,1) = w0;
    for trial_no = 2:num_trials
        delta = r(trial_no-1) - u(:,trial_no-1)' * w(:,trial_no-1);
        w(:,trial_no) = w(:,trial_no-1) + espsilon*delta*u(:,trial_no-1);
    end
end
