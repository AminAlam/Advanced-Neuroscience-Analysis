%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RW rule 
% Extinction
clc
clear
close all
save_figures = 1;

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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/extinction','-dpng','-r0')
end
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
title('Partial')
legend('\alpha = 0.1', '\alpha = 0.4', '\alpha = 0.7')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/partial','-dpng','-r0')
end
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

legend('$\omega_1$', '$\omega_2$', 'interpreter','LaTex')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/blocking','-dpng','-r0')
end
%% inhibitory
clc
close all
figure
num_trials = 200;
espsilon = 5e-2;
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
legend('$\omega_1$', '$\omega_2$', 'interpreter','LaTex')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/inhibitory','-dpng','-r0')
end
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
legend('$\omega_1$', '$\omega_2$', 'interpreter','LaTex')
ylim([0, 1])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/overshadow_1','-dpng','-r0')
end

figure
num_trials = 200;
espsilon = 5e-2;
alpha1 = 0.9;
alpha2 = 0.1;
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
title('Overshadow | \alpha_1 = 0.9 , \alpha_2 = 0.1')
legend('$\omega_1$', '$\omega_2$', 'interpreter','LaTex')
ylim([0, 1])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/overshadow_2','-dpng','-r0')
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kalman Filter
clc
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
legend('$\omega_1$', '$\omega_2$', 'interpreter','LaTex')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/drift','-dpng','-r0')
end


% blocking
figure
num_trials = 20;
u = ones(2, num_trials);
r = ones(1, num_trials);
u(2, 1:end/2) = 0;
w0 = [0;0];
tau = 0.5;
cov0 = [0.6 0;0, 0.6];
noise_p = [0.01 0; 0, 0.01];
[w, cov_mat] = kalamFilter(u, r, cov0, w0, tau, noise_p, num_trials);
subplot(2,1,1)
plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
plot(1:num_trials, w(2,:), '--k', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('\omega')
legend('$\omega_1$', '$\omega_2$', 'interpreter','LaTex')
title('Blocking - mean')

subplot(2,1,2)
plot(1:num_trials, squeeze(cov_mat(1,1,:)), 'k', 'LineWidth', 2)
hold on
plot(floor(num_trials/2)+1:num_trials, squeeze(cov_mat(2,2,floor(num_trials/2)+1:end)), '--k', 'LineWidth', 2)
xline(floor(num_trials/2)+1, '--k');
ylim([0, 1]);
xlim([1, num_trials])
xlabel('Trial Number')
ylabel('$\sigma^2$', 'interpreter','LaTex')
legend('$\sigma_1^2$', '$\sigma_2^2$', 'interpreter','LaTex')
title('Blocking - variance')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/blocking2','-dpng','-r0')
end

% unblocking
figure
num_trials = 20;
u = ones(2, num_trials);
r = ones(1, num_trials);
r(end/2+1:end) = 2;
u(2, 1:end/2) = 0;
w0 = [0;0];
tau = 0.5;
cov0 = [0.6 0; 0, 0.6];
noise_p = [0.01 0; 0, 0.01];
[w, cov_mat] = kalamFilter(u, r, cov0, w0, tau, noise_p, num_trials);
subplot(2,1,1)
plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
plot(1:num_trials, w(2,:), '--k', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('\omega')
legend('$\omega_1$', '$\omega_2$', 'interpreter','LaTex')
title('unBlocking - mean')

subplot(2,1,2)
plot(1:num_trials, squeeze(cov_mat(1,1,:)), 'k', 'LineWidth', 2)
hold on
plot(floor(num_trials/2)+1:num_trials, squeeze(cov_mat(2,2,floor(num_trials/2)+1:end)), '--k', 'LineWidth', 2)
xline(floor(num_trials/2)+1, '--k');
ylim([0, 1]);
xlim([1, num_trials])
xlabel('Trial Number')
ylabel('$\sigma^2$', 'interpreter','LaTex')
legend('$\sigma_1^2$', '$\sigma_2^2$', 'interpreter','LaTex')
title('unBlocking - variance')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/unblocking','-dpng','-r0')
end

% backward blocking
figure
num_trials = 20;
u = ones(2, num_trials);
r = ones(1, num_trials);
u(2, end/2+1:end) = 0;
w0 = [0;0];
tau = 0.5;
cov0 = [0.6 0;0, 0.6];
noise_p = [0.01 0; 0, 0.01];
[w, cov_mat] = kalamFilter(u, r, cov0, w0, tau, noise_p, num_trials);
subplot(2,1,1)
plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
plot(1:num_trials, w(2,:), '--k', 'LineWidth', 2)
xlabel('Trial Number')
ylabel('\omega')
legend('$\omega_1$', '$\omega_2$', 'interpreter','LaTex')
title('Backward Blocking - mean')

subplot(2,1,2)
plot(1:num_trials, squeeze(cov_mat(1,1,:)), 'k', 'LineWidth', 2)
hold on
plot(floor(num_trials/2)+1:num_trials, squeeze(cov_mat(2,2,floor(num_trials/2)+1:end)), '--k', 'LineWidth', 2)
xline(floor(num_trials/2)+1, '--k');
ylim([0, 1]);
xlim([1, num_trials])
xlabel('Trial Number')
ylabel('$\sigma^2$', 'interpreter','LaTex')
legend('$\sigma_1^2$', '$\sigma_2^2$', 'interpreter','LaTex')
title('Backward Blocking - variance')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/backward_blocking','-dpng','-r0')
end
%% paper contour plots
clc
close all
% backward blocking
num_trials = 20;
u = ones(2, num_trials);
r = ones(1, num_trials);
u(2, end/2+1:end) = 0;
w0 = [0;0];
tau = 1.2;
cov0 = [0.6 0; 0, 0.6];
noise_p = [0.02 0; 0, 0.02];
[w, cov_mat] = kalamFilter(u, r, cov0, w0, tau, noise_p, num_trials);

r_list = 0.5:0.2:3;
theta = 0:0.01:2*pi;
colors = linspace(1, 0, length(r_list));
i = 1;

for t = [1, 9, 19]
    figure
    patch(box,boxy,'k','FaceAlpha',1)
    hold on
    cov_mat_c = cov_mat(:,:,t);
    w_c = w(:, t);
    j = length(r_list);
    for r = flip(r_list)
        x = sin(theta)*r;
        y = cos(theta)*r;
        tmp = cov_mat_c*[x;y];
        x = w_c(1) + tmp(1, :);
        y = w_c(2) + tmp(2, :);
        fill(x, y, [1, 1, 1]*colors(j), 'LineStyle', 'none')
        j = j-1;
    end
    scatter(w_c(1), w_c(2), '*k', 'LineWidth', 1)
    generated_data = mvnrnd(w_c,cov_mat_c, 500);
    scatter(generated_data(:,1), generated_data(:,2), '.b')
    xlim([-1, 2])
    ylim([-1, 2])
    xlabel('$\overline {\omega_1}$ ', 'interpreter' ,'LaTex', 'FontSize', 16)
    ylabel('$\overline {\omega_2}$ ', 'interpreter' ,'LaTex', 'FontSize', 16)
    i = i +1;
    box = [-1 -1 2 2];
    boxy = [2 -1 -1 2];
end
%% effect of process noise and measurement noise
clc
close all

taus = [0.2 , 0.5, 1];
noise_p_coeef = [0.01, 0.1, 1];

for k = 1:3
    for kk = 1:3
        tau = taus(1, k);
        noise_p = eye(2)*noise_p_coeef(1, kk);
        % backward blocking
        figure
        num_trials = 20;
        u = ones(2, num_trials);
        r = ones(1, num_trials);
        u(2, end/2+1:end) = 0;
        w0 = [0;0];
        cov0 = [0.6 0;0, 0.6];
        [w, cov_mat] = kalamFilter(u, r, cov0, w0, tau, noise_p, num_trials);
        subplot(2,1,1)
        plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
        hold on
        plot(1:num_trials, w(2,:), '--k', 'LineWidth', 2)
        xlabel('Trial Number')
        ylabel('\omega')
        legend('$\omega_1$', '$\omega_2$', 'interpreter','LaTex')
        title("Backward Blocking - mean - mNoise "+num2str(tau)+" | "+"Pnoise "+num2str(noise_p_coeef(1, kk)))

        subplot(2,1,2)
        plot(1:num_trials, squeeze(cov_mat(1,1,:)), 'k', 'LineWidth', 2)
        hold on
        plot(floor(num_trials/2)+1:num_trials, squeeze(cov_mat(2,2,floor(num_trials/2)+1:end)), '--k', 'LineWidth', 2)
        xline(floor(num_trials/2)+1, '--k');
        ylim([0, 1]);
        xlim([1, num_trials])
        xlabel('Trial Number')
        ylabel('$\sigma^2$', 'interpreter','LaTex')
        legend('$\sigma_1^2$', '$\sigma_2^2$', 'interpreter','LaTex')
        title("Backward Blocking - variance - mNoise "+num2str(tau)+" | "+"Pnoise "+num2str(noise_p_coeef(1, kk)))

        if save_figures
            set(gcf,'PaperPositionMode','auto')
            print("Report/images/noise_effect"+num2str(k)+"_"+num2str(kk),'-dpng','-r0')
        end

        r_list = 0.5:0.2:3;
        theta = 0:0.01:2*pi;
        colors = linspace(1, 0, length(r_list));
        i = 1;
        for t = [1, 9, 19]
            figure
            patch(box,boxy,'k','FaceAlpha',1)
            hold on
            cov_mat_c = cov_mat(:,:,t);
            w_c = w(:, t);
            j = length(r_list);
            for r = flip(r_list)
                x = sin(theta)*r;
                y = cos(theta)*r;
                tmp = cov_mat_c*[x;y];
                x = w_c(1) + tmp(1, :);
                y = w_c(2) + tmp(2, :);
                fill(x, y, [1, 1, 1]*colors(j), 'LineStyle', 'none')
                j = j-1;
            end
            scatter(w_c(1), w_c(2), '*k', 'LineWidth', 1)
            generated_data = mvnrnd(w_c,cov_mat_c, 500);
            scatter(generated_data(:,1), generated_data(:,2), '.b')
            xlim([-1, 2])
            ylim([-1, 2])
            xlabel('$\overline {\omega_1}$ ', 'interpreter' ,'LaTex', 'FontSize', 16)
            ylabel('$\overline {\omega_2}$ ', 'interpreter' ,'LaTex', 'FontSize', 16)
            i = i +1;
            if save_figures
                set(gcf,'PaperPositionMode','auto')
                print("Report/images/noise_effect"+num2str(t)+"_"+num2str(k)+"_"+num2str(kk),'-dpng','-r0')
            end
        end
        disp("k "+num2str(k) + " | kk "+num2str(kk))
        tau
        noise_p_coeef(1, kk)
    end
end

%% first s1 -> 'r' then s1 -> '-r'
clc
close all

figure
num_trials = 20;
u = ones(2, num_trials);
u(2, :) = 0;
r = ones(1, num_trials);
r(end/2+1:end) = -1;
w0 = [0;0];
tau = 1;
cov0 = [0.6 0;0, 0.6];
noise_p = [0.01 0; 0, 0.01];
[w, cov_mat] = kalamFilter(u, r, cov0, w0, tau, noise_p, num_trials);

subplot(2,1,1)
plot(1:num_trials, w(1,:), 'k', 'LineWidth', 2)
hold on
xline(floor(num_trials/2)+1, '--k');
xlim([1, num_trials])
xlabel('Trial Number')
ylabel('\omega', 'FontSize', 16)
legend('$\omega_1$', 'interpreter','LaTex')
title('mean')

subplot(2,1,2)
plot(1:num_trials, squeeze(cov_mat(1,1,:)), 'k', 'LineWidth', 2)
xline(floor(num_trials/2)+1, '--k');
ylim([0, 1]);
xlim([1, num_trials])
xlabel('Trial Number')
ylabel('$\sigma^2$', 'interpreter','LaTex', 'FontSize', 16)
legend('$\sigma_1^2$', 'interpreter','LaTex')
title('variance')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Report/images/pos_neg','-dpng','-r0')
end
%% figure 3 of the paper
clc
clear
close all


figure
num_trials = 100;
u = ones(1, num_trials);
v = normrnd(0,0.1,1,num_trials);
v_r = normrnd(0,0.2,1,num_trials);
c = zeros(1, num_trials);
c(40) = 1;
c(90) = 1;
phi = zeros(1,num_trials);
phi(40) = -2;
phi(90) = 6;

w_real = zeros(1, num_trials);
r = zeros(1, num_trials);
r(:,1) = w_real(:,1) + v_r(:,1) + c(:,1)*phi(:,1);
for trial_no = 2:num_trials
    w_real(:,trial_no) = w_real(:,trial_no-1) + v(:, trial_no-1) + c(:, trial_no-1)*phi(:, trial_no-1);
    r(:,trial_no) = w_real(:,trial_no) + v_r(:,trial_no) + c(:,trial_no)*phi(:,trial_no);
end

% kalman predict w
w0 = 0;
tau = 0.6;
cov0 = 0.6;
noise_p = 0.01;
phi_noise = 100;
gamma = 3.3;

[w, cov_mat, B] = kalamFilterInhanced(u, r, cov0, w0, tau, gamma, noise_p, phi_noise, num_trials);

scatter(1:num_trials, r, 'xk')
hold on
scatter(1:num_trials, w_real, '.k')
scatter(0, 0, 400, 'k', 'LineWidth', 2)
scatter(40, -2, 400, 'k', 'LineWidth', 2)
scatter(90, 4, 400, 'k', 'LineWidth', 2)
legend('$r(t)$', '$w(t)$', 'interpreter','LaTex', 'FontSize', 16,'Location','northwest')
xlabel('Trial Number')
ylabel('$\omega$', 'interpreter','LaTex', 'FontSize', 16)
title('mean')
ylim([-4 5])

figure
scatter(1:num_trials, w, 'ok')
hold on
scatter(1:num_trials, r, 'xk')
scatter(1:num_trials, w_real, '.k')
legend('$\hat{w}(t)$', 'interpreter','LaTex', 'FontSize', 16,'Location','northwest')
xlabel('Trial Number')
ylabel('$\omega$', 'interpreter','LaTex', 'FontSize', 16)
title('mean')
ylim([-4 5])

figure

plot(1:num_trials, squeeze(cov_mat(1,1,:)),'--k','LineWidth',1.5)
hold on
plot(1:num_trials, B, 'k','LineWidth',1.5)
hold on
yline(gamma,'-.k','LineWidth',1.5);
ylim([0 10])
legend("ACh","NE","$gamma$",'Interpreter','LaTex','location','northwest')
xlabel("t")

%% MSE
clc
clear
close all
gammas = 0:0.5:30;
SSE_all = zeros(100, size(gammas, 2));

num_trials = 100;
u = ones(1, num_trials);

w0 = 0;
tau = 0.7;
cov0 = 0.6;
noise_p = 0.01;
phi_noise = 100;

for j = 1:100
    
    v = normrnd(0,0.1,1,num_trials);
    v_r = normrnd(0,0.5,1,num_trials);
    c = zeros(1, num_trials);
    c(40) = 1;
    c(90) = 1;
    phi = zeros(1,num_trials);
    phi(40) = -2;
    phi(90) = 4;

    w_real = zeros(1, num_trials);
    r = zeros(1, num_trials);
    r(:,1) = w_real(:,1) + v_r(:,1) + c(:,1)*phi(:,1);
    for trial_no = 2:num_trials
        w_real(:,trial_no) = w_real(:,trial_no-1) + v(:, trial_no-1) + c(:, trial_no-1)*phi(:, trial_no-1);
        r(:,trial_no) = w_real(:,trial_no) + v_r(:,trial_no) + c(:,trial_no)*phi(:,trial_no);
    end

    i = 1;
    MSEs = zeros(size(gammas));
    for gamma = gammas
        [w, cov_mat, B] = kalamFilterInhanced(u, r, cov0, w0, tau, gamma, noise_p, phi_noise, num_trials);
        MSEs(i) = sse(w, w_real);
        i = i + 1;
    end
    
    SSE_all(j, :) = MSEs;

end
%%
figure
plot(gammas, mean(SSE_all, 1), 'k', 'LineWidth', 2)
hold on
errorbar(gammas, mean(SSE_all, 1), sqrt(var(SSE_all, 1, 1))/10, 'k')
xlabel("$\gamma$",'interpreter','LaTex')
ylabel("MSE")
