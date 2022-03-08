%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Integrate and Fire
clc
clear
close all
save_figures = 1;

% %%%%%%%%%%%%%%%%%%%%%%%% part a
% generating spikes
syms x;
dt = 1e-3;
r = 100;
%% from uniform dist
clc
t_0 = 0;
t_end = 20;
rf_period = 0;
spike_times = poissonFromUnifrom(t_0, t_end, dt, r, rf_period);

for i = spike_times(1:50)
    plot([i-1,i-1],[0,0.5],'black','LineWidth',2)
    hold on
end

ylim([0 0.75])
xlabel("times")
title("spikes")

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q1a_uniform','-dpng','-r0')
end
%% from exponential dist
t_0 = 0;
t_end = 10;
rf_period = 0;
spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);

for i = spike_times(1:50)
    plot([i-1,i-1],[0,0.5],'black','LineWidth',2)
    hold on
end

ylim([0 0.75])
xlabel("times")
title("spikes")

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q1a_poisson','-dpng','-r0')
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%% part b
% Spikes counts probability histogram

spike_counts = [];
t_end = 1;
for i = 1:1000
    spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);
    spike_counts = [spike_counts, length(spike_times)];
end

histogram(spike_counts, 'Normalization', 'pdf')

poiss_dist = @(x) r.^x./factorial(x).*exp(-r);
hold on
counts = 0:200;
plot(counts, poiss_dist(counts), 'LineWidth', 2)
title('Spikes count probability histogram')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q1b_uniform','-dpng','-r0')
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%% part c
% ISIs histogram
clc
t_end = 100;
spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);
spike_times_diff = diff(spike_times);

histogram(spike_times_diff, 'Normalization','pdf')
title("ISI histogram")
hold on
times = 0:0.001:0.09;

exp_dist = @(x) r*exp(-r*x);
plot(times, exp_dist(times), 'LineWidth', 2)

xlim([times(1), times(end)])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q1c_uniform','-dpng','-r0')
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%% renewal process
k = 5;
% part b
spike_counts = [];
t_end = 1;
for i = 1:1000
    spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);
    spike_times_ds = downsample(spike_times, k);
    spike_counts = [spike_counts, length(spike_times_ds)];
end

histogram(spike_counts, 'Normalization', 'pdf')

poiss_dist = @(x) r.^x./factorial(x).*exp(-r);
hold on
counts = 0:200;
plot(counts, poiss_dist(counts), 'LineWidth', 2)
title('Spikes count probability histogram')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q1c_r1','-dpng','-r0')
end

% part c
figure
t_end = 400;
spike_times = 0;

while spike_times(end)<t_end
    t = exprnd(1/r,1,1);
    spike_times = [spike_times, spike_times(end)+t];
end
spike_times_ds = renewalProcess(spike_times, k);
spike_times_diff = diff(spike_times_ds);

histogram(spike_times_diff, 'Normalization','pdf')
title("ISI Histogram")
hold on
times = 0:0.001:0.09;

exp_dist = @(x) r*exp(-r*x);
plot(times, exp_dist(times), 'LineWidth', 2)

xlim([times(1), times(end)])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q1c_r2','-dpng','-r0')
end
%% part d
CV = cvCalculator(diff(spike_times_ds))
%% %%%%%%%%%%%%%%%%%%%%%%%%%% Effect of the refractory period
% part g
CV1s = [];
CV4s = [];
CV51s = [];

dts = 1e-4:1e-4:30e-3;
t_0 = 0;
t_end = 40;

for rf_period = [0, 1e-3]
    CV1 = [];
    CV4 = [];
    CV51 = [];
    for dt = dts
        r = 1/dt;
        spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);
        
        k = 1;
        spike_times_ds = renewalProcess(spike_times, k);
        spike_times_ds = diff(spike_times_ds);
        CV = cvCalculator(spike_times_ds);
        CV1 = [CV1, CV];
        
        k = 4;
        spike_times_ds = renewalProcess(spike_times, k);
        spike_times_ds = diff(spike_times_ds);
        CV = cvCalculator(spike_times_ds);
        CV4 = [CV4, CV];

        k = 51;
        spike_times_ds = renewalProcess(spike_times, k);
        spike_times_ds = diff(spike_times_ds);
        CV = cvCalculator(spike_times_ds);
        CV51 = [CV51, CV];
    end
    CV1s = [CV1s; CV1];
    CV4s = [CV4s; CV4];
    CV51s = [CV51s; CV51];
end

figure
hold on
plot(dts, CV1s(1,:), 'r--', 'LineWidth', 2)
plot(dts, CV4s(1,:), 'r--', 'LineWidth', 2)
plot(dts, CV51s(1,:), 'r--', 'LineWidth', 2)

plot(dts, CV1s(2,:), 'k', 'LineWidth', 2)
plot(dts, CV4s(2,:), 'k', 'LineWidth', 2)
plot(dts, CV51s(2,:), 'k', 'LineWidth', 2)
ylim([0, 1.1])

xlabel('$\bar{dt}$', 'Interpreter','latex', 'FontSize', 18)
ylabel('$C_v$', 'Interpreter','latex', 'FontSize', 18)

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q1g','-dpng','-r0')
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Leaky Integrate and Fire
clc
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%% stimulation of leaky integrate and fire
% part a
R = 1;
I = 20e-3;
Vr = 0;
Vth = 15e-3;
Vsp = 70e-3;
t_0 = 0;
t_end = 100e-3;
tau = 10e-3;
dt = 1e-4;
V = Vr;
times = t_0:dt:t_end;

f = @(V_st) (-V_st+R*I)*dt/tau;
for t = times
    if V(end)>Vth
        V(end) = Vsp;
        V_st = 0;
    else
        V_st = V(end);
    end
    
    dV = f(V_st);
    V = [V, V_st+dV];
end

plot(times, V(2:end), 'LineWidth', 2)
ylim([0, 100e-3])
xlabel('time(s)')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q2a','-dpng','-r0')
end
%% part c
clc
close all
Vr = 0;
V = Vr;
t_0 = 0;
dt = 1e-4;
t_end = 1000e-3;
rf_period = 0;
Vsp = 70e-3;
Vth = 15e-3;
tau = 5e-3;
tau_peak = 1.5e-3;
r = 100;

times = t_0:dt:t_end;
spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);
% spike_times = poissonFromUnifrom(t_0, t_end, dt, r, rf_period);
spikes = times*0;

for i = 1:length(spike_times)
    [~, col] = find(abs(times-spike_times(i))==min(abs(times-spike_times(i))));
    spikes(col) = 1;
end

t_kernel = 0:dt:10*tau_peak;
kernel = @(t) t.*exp(-t/tau_peak);
kernel_norm = max(kernel(t_kernel));
kernel = kernel(t_kernel)/kernel_norm;

I = 20e-3*conv(spikes, kernel, 'same');

figure
plot(t_kernel, kernel, 'LineWidth', 2)
xlabel('time (ms)')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q2c_11','-dpng','-r0')
end

figure
subplot(2,1,1)
plot(times, I, 'LineWidth', 1)
hold on
for i = spike_times
    plot([i,i],[0,1e-2],'black','LineWidth',1)
end

xlabel('time (s)')
title("Current")
legend("Current", "Spikes")

V = LIF(times, Vr, I, dt, tau, Vth, Vsp);

subplot(2,1,2)
plot(times, V(2:end), 'LineWidth', 2)
xlabel('time (s)')
title("Voltage")

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q2c_12','-dpng','-r0')
end

[~, col] = find(V>=Vsp);
spike_times = times(col);
CV = cvCalculator(diff(spike_times))
%% part c - figure 8 of the paper
clc
close all
rf_period = 0;
dt = 1e-4;
t_end = 30;
times = t_0:dt:t_end;
taus = [0.1:0.2:1 , 2:2:12] * 10^-3;
r = 200;
ks = [1:10, 20:20:100];

t_kernel = dt:dt:20*tau_peak;
kernel = @(t) t.*exp(-t/tau_peak);
kernel_norm = max(kernel(t_kernel));
kernel = kernel(t_kernel)/kernel_norm;

CVs = zeros(length(taus), length(ks));

k_i = 0;
for k = ks
    k_i = k_i+1
    tau_i = 0;
    for tau = taus
        tau_i = tau_i+1;
        spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);
        spikes = times*0;

        for i = 1:length(spike_times)
            [~, col] = find(abs(times-spike_times(i))==min(abs(times-spike_times(i))));
            spikes(col) = 1;
        end
        
        I = 20*conv(spikes, kernel, 'same');
        V = LIF(times, Vr, I, dt, tau, Vth, Vsp);
        [~, col] = find(V>=Vsp);
        spike_times = times(col);
        spike_times_ds = renewalProcess(spike_times, k);
        CV = cvCalculator(diff(spike_times_ds));
        CVs(tau_i, k_i) = CV;
    end
end

figure
for cv = 0.1:0.1:0.8
    [row, col] = find(CVs<cv+0.05 & CVs>cv-0.05);
    tau_selected = taus(row);
    k_selected = ks(col);
    loglog(k_selected, tau_selected*1000, 'LineWidth', 1);
    hold on
end
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Tau')
ylabel('Nth')
legend('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q2c2_arsalan','-dpng','-r0')
end
%% part c - effect of magnitude and width of the kernel
clc
close all

dt = 1e-4;
t_end = 1;
r = 200;
times = t_0:dt:t_end;

t_peaks = (1.5:0.5:20)*1e-3;
mags = (20:10:100)*1e-3;
CVs = zeros(length(t_peaks), length(mags));

spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);
spikes = times*0;

for i = 1:length(spike_times)
    [~, col] = find(abs(times-spike_times(i))==min(abs(times-spike_times(i))));
    spikes(col) = 1;
end

t_peak_i = 0;
for tau_peak = t_peaks
    t_peak_i = t_peak_i+1
    mag_i = 0;
    for mag = mags
        mag_i = mag_i+1;
        t_kernel = dt:dt:10*tau_peak;
        kernel = @(t) t.*exp(-t/tau_peak);
        kernel_norm = max(kernel(t_kernel));
        kernel = kernel(t_kernel)/kernel_norm;

        I = mag*conv(spikes, kernel, 'same');
        V = LIF(times, Vr, I, dt, tau, Vth, Vsp);
        [~, col] = find(V>=Vsp);
        spike_times_ds = times(col);
        CV = cvCalculator(diff(spike_times_ds));
        CVs(t_peak_i, mag_i) = CV;
    end
end

figure
imagesc(t_peaks, mags, CVs)
colorbar
xlabel('magnitude')
ylabel('tau peak')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q2c3','-dpng','-r0')
end
%% part d
clc
close all
tau_peak = 1.5e-3;
t_0 = 0;
t_end = 5;
Vr = 0;
tau = 10e-3;
Vsp = 70e-3;
Vth = 15e-3;
dt = 10e-4;
rf_period = 0;
num_neurons = 100;
r = 100;
inhib_percents = (1:1:60)/100;
CVs = zeros(1, length(inhib_percents));

times = t_0:dt:t_end;
spikes_all = [];

t_kernel = dt:dt:10*tau_peak;
kernel = @(t) t.*exp(-t/tau_peak);
kernel_norm = max(kernel(t_kernel));
kernel = kernel(t_kernel)/kernel_norm;

for neuron = 1:num_neurons
    spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);
    spike_times_all.(sprintf("N%i",neuron)).spike_times = spike_times;
    spikes = times*0;
    for i = 1:length(spike_times)
        [~, col] = find(abs(times-spike_times(i))==min(abs(times-spike_times(i))));
        spikes(col) = 1;
    end
    spike_times_all.(sprintf("N%i",neuron)).spikes = spikes;
end

inhib_i = 0;
for inhib_perc = inhib_percents
    inhib_i = inhib_i+1;
    spikes_inh_exc = spikes_all;
    
    spike_times_ds_all = [];
    I = 0;
    for  neuron = 1:num_neurons
        spikes = spike_times_all.(sprintf("N%i",neuron)).spikes;
        if rand(1)<inhib_perc
            I = I-20e-3*conv(spikes, kernel, 'same');
        else
            I = I+20e-3*conv(spikes, kernel, 'same');
        end
    end
     V = LIF(times, Vr, I, dt, tau, Vth, Vsp);
    [~, col] = find(V>=Vsp);
    spike_times_ds = times(col);
    spike_times_ds_all = [spike_times_ds_all, diff(spike_times_ds)];
    CV = cvCalculator(spike_times_ds_all);
    CVs(inhib_i) = CV;
end
plot(inhib_percents, CVs, 'k', 'LineWidth', 2)
xlabel('Percentage of Inhibitory Neurons')
ylabel('CV')
xlim([0, 0.4])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q2d','-dpng','-r0')
end
%% part e
clc
close all
t_0 = 0;
t_end = 1;
rf_period = 0;
r = 100;
num_neurons = 100;
spike_times_all = [];

for neuron = 1:num_neurons
    spike_times = poissonFromExpoential(t_0, t_end, r, rf_period);
    spike_times_all.(sprintf("N%i",neuron)) = spike_times;
end

N_Ms = 0.1:0.1:0.9;
Ds = [5:5:100, 200:50:400]*1e-3;
t_win_step = 10e-3;
CVs = zeros(length(N_Ms), length(Ds));
N_M_i = 0;

for N_M = N_Ms
    N_M_i = N_M_i+1
    D_i = 0;
    for D = Ds
        out_spike_times = [];
        D_i = D_i+1;
        t_win = 0;
        while t_win<spike_times(end)
            num_spikes = 0;
            time_spikes_tmp = [];
            for neuron = 1:num_neurons
                spike_times = spike_times_all.(sprintf("N%i",neuron));
                [~, col] = find(spike_times>=t_win & spike_times<t_win+D);

                if ~isempty(col)
                    num_spikes = num_spikes+1;
                    time_spikes_tmp = [time_spikes_tmp, spike_times(col(end))];
                end

                if num_spikes>=num_neurons*N_M
                    out_spike_times = [out_spike_times, time_spikes_tmp(end)];
                    break
                end
            end
            t_win = t_win+t_win_step;
        end
        CV = cvCalculator(diff(out_spike_times));
        CVs(N_M_i, D_i) = CV;
    end
end


figure
imagesc(N_M, Ds, CVs)
colorbar
xlabel('N/M')
ylabel('D')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print('Q2e','-dpng','-r0')
end
%% Functions

function spike_times = poissonFromUnifrom(t_0, t_end, dt, r, rf_period)
    interval = t_0:dt:t_end;
    spikes = rand(1,length(interval)) < (interval*0+1)*dt*r;
    spike_times = interval(find(spikes.*interval));
    spike_times_diff = diff(spike_times);
    [~, col] = find(spike_times_diff>rf_period);
    spike_times = spike_times(col+1);
end

function spike_times = poissonFromExpoential(t_0, t_end, r, rf_period)
    spike_times = t_0;
    while 1
        t = exprnd(1/r,1,1);
        if spike_times(end)+t+rf_period > t_end
            break
        end
        spike_times = [spike_times, spike_times(end)+t+rf_period];
    end
end

function CV = cvCalculator(spike_times)
    std_data = std(spike_times, 0, 'all');
    mean_data = mean(spike_times, 'all');
    CV = std_data/mean_data;
end

function spike_times_ds = renewalProcess(spike_times, k)
    spike_times_ds = downsample(spike_times, k);
end

function V = LIF(times, V, I, dt, tau, Vth, Vsp)
    for t = 1:length(times)
        if V(end)>Vth
            V(end) = Vsp;
            V_st = 0;
        else
            V_st = V(end);
        end
        dV = (-V_st+I(t))*dt/tau;
        V = [V, V_st+dV];
    end
end