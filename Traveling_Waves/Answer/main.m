clc
clear
close all
load("data/ArrayData.mat")
load("data/CleanTrials.mat")
fs = 200;

num_channels = numel(chan);
% removing bad trials
for ch_no = 1:num_channels
    chan(ch_no).lfp = chan(ch_no).lfp(:, Intersect_Clean_Trials);
end
num_trials = size(chan(1).lfp, 2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LFP analysis
%% part a - finding the most dominant frequency oscillation using 
Ps = 0;
for ch_no = 1:num_channels
    lfp_data = chan(ch_no).lfp;
    for trial_no = 1:num_trials
        trial_data = lfp_data(:, trial_no);
        trial_data = zscore(trial_data);
        m = length(trial_data);
        n = pow2(nextpow2(m));
        Y = fft(trial_data, n);
        Y = fftshift(Y);
        Ps = Ps+abs(Y);
    end
end

figure
f = (-n/2:n/2-1)*(fs/n);
Ps_plot = 10*log10(Ps.^2/n);

p = polyfit(log10(f(n/2+2:end)),Ps_plot(n/2+2:end)',1);

pink_noise = polyval(p,log10(f(n/2+2:end)))';
semilogx(f(n/2+2:end), pink_noise, '--r', 'LineWidth', 2)
hold on
semilogx(f(n/2+2:end), Ps_plot(n/2+2:end), 'k', 'LineWidth', 2)
title('Averaged power spectrum of all trials of all channels')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
xlim([1, 100])
legend('Estimated Pink Noise')
grid on
% removing pink noise
figure
hold on
P_smooth = movmean(Ps_plot(n/2+2:end)-pink_noise, 10); % smoothing the powerspectrum
plot(f(n/2+2:end), P_smooth, 'r', 'LineWidth', 2) 

plot(f(n/2+2:end), Ps_plot(n/2+2:end)-pink_noise, 'k', 'LineWidth', 2) 

legend('Smoothed Power Spectrum', 'Power Spectrum')
title('Averaged power spectrum of all trials of all channels (pink noise removed)')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
xlim([1, 70])
grid on
%% part b - clustering electrodes based on their dominant oscillation frequency
figure
hold on

for ch_no = 1:num_channels
    lfp_data = chan(ch_no).lfp;
    Ps = 0;
    for trial_no = 1:num_trials
        trial_data = lfp_data(:, trial_no);
        trial_data = zscore(trial_data);
        m = length(trial_data);
        n = pow2(nextpow2(m));
        Y = fft(trial_data, n);
        Y = fftshift(Y);
        Ps = Ps+abs(Y);
    end
    f = (-n/2:n/2-1)*(fs/n);
    Ps_plot = removePinkNoise(Ps, f, n);
    plot(f(n/2+2:end), Ps_plot(n/2+2:end))
    
end

%% 
stft_map = 0;
for ch_no = 1:num_channels
    lfp_data = chan(ch_no).lfp;
    for trial_no = 1:num_trials
        trial_data = lfp_data(:, trial_no);
        [s,f,t] = stft(trial_data,fs,'Window',kaiser(20,5),'OverlapLength',10,'FFTLength',fs);
        stft_map = stft_map + abs(s); 
    end
end
%%
stft_map_disp = stft_map/(num_channels*num_trials);
stft_map_disp = sqrt(stft_map_disp);
t_disp = linspace(-1.2, 3, length(t));
pink_noise = (1./f)';
pink_noise = repmat(pink_noise', [1, length(t_disp)]);

stft_map_disp = stft_map_disp-pink_noise;
pcolor(t_disp, f, stft_map_disp)
ylabel('Frequency (Hz)')
xlabel('Time (s)')
ylim([1, 30])
colormap jet
title('Average STFT of all trials of all channels')
shading interp


%% Functions

function Ps_plot = removePinkNoise(Ps, f, n)
    Ps_plot = 10*log10(Ps.^2/n);
    p = polyfit(log10(f(n/2+2:end)),Ps_plot(n/2+2:end)',1);
    pink_noise = polyval(p,log10(f(n/2+2:end)))';
    Ps_plot(n/2+2:end) = Ps_plot(n/2+2:end)-pink_noise;
end