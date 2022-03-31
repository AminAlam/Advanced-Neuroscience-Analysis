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
    lfp_data = zscore(lfp_data);
    for trial_no = 1:num_trials
        trial_data = lfp_data(:, trial_no);
        m = length(trial_data);
        n = pow2(nextpow2(m));
        Y = fft(trial_data, n);
        Y = fftshift(Y);
        Ps = Ps+abs(Y);
    end
end
normalize_constant = 10*log10((num_channels*num_trials)^2);

f = (-n/2:n/2-1)*(fs/n);
Ps_plot = 10*log10(Ps.^2/n);
pink_noise = 1./f(n/2+2:end);
[~,~,spectrum_regressed] = regress(Ps_plot(n/2+2:end), pink_noise');
pink_spectrum = Ps_plot(n/2+2:end) - spectrum_regressed;

figure
loglog(f(n/2+2:end), pink_spectrum, '--r', 'LineWidth', 2)
hold on
loglog(f(n/2+2:end), Ps_plot(n/2+2:end), 'k', 'LineWidth', 2)
title('Averaged power spectrum of all trials of all channels (not normalized)')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
xlim([1, 100])
legend('Estimated Pink Noise')
grid on

figure
plot(f(n/2+2:end), pink_spectrum-normalize_constant, '--r', 'LineWidth', 2)
hold on
plot(f(n/2+2:end), Ps_plot(n/2+2:end)-normalize_constant, 'k', 'LineWidth', 2)
title('Averaged power spectrum of all trials of all channels')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
xlim([1, 100])
legend('Estimated Pink Noise')
grid on

% removing pink noise
figure
hold on
spectrum_clean = Ps_plot(n/2+2:end) - pink_spectrum;
plot(f(n/2+2:end), Ps_plot(n/2+2:end)-normalize_constant, 'r', 'LineWidth', 2) 

plot(f(n/2+2:end), spectrum_clean-normalize_constant, 'k', 'LineWidth', 2) 

legend('Original', 'Denoised (No Pink Noise)')
title('Averaged power spectrum of all trials of all channels')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
xlim([1, 70])
grid on
%% part b - clustering electrodes based on their dominant oscillation frequency
clc
close all

figure
hold on
dominant_freq_mat = ChannelPosition*nan;
normalize_constant = 10*log10((num_trials)^2);
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
    Ps = 10*log10(Ps.^2/n);
    Ps_plot = removePinkNoise(Ps, f, n, 1);
    plot(f(n/2+2:end), Ps_plot(n/2+2:end)-normalize_constant)
    [row, ~] = find(Ps_plot(n/2+2:end) == max(Ps_plot(n/2+2:end)));
    f_tmp = f(n/2+2:end);
    dominant_freq = f_tmp(row);
    [row, col] = find(ChannelPosition==ch_no);
    dominant_freq_mat(row, col) = dominant_freq;
end

title('Averaged power spectrum over all trials of each channel')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
xlim([0, 70])
ylim([-35, 10])
grid on

figure
plt = imagesc(dominant_freq_mat);
set(plt,'AlphaData', ~isnan(dominant_freq_mat))
colormap jet
c_bar = colorbar;
caxis([0, 13])
title('Dominant Frequencies')
ylabel(c_bar,'Frequency (Hz)')
%% part c - time-frequncy analysis of the LFP data
clc
close all
% STFT Method
stft_map = 0;
for ch_no = 1:num_channels
    lfp_data = chan(ch_no).lfp;
    lfp_data = zscore(lfp_data);
    for trial_no = 1:num_trials
        trial_data = lfp_data(:, trial_no);
        [s,f,time_stft] = stft(trial_data,fs,'Window',kaiser(60,5),'OverlapLength',40,'FFTLength',fs);
        stft_map = stft_map + abs(s); 
    end
end

stft_map_tmp = [];

for t = 1:size(stft_map, 2)
    Ps = stft_map(:, t);
    n = length(Ps);
    Ps_plot = removePinkNoise(Ps, f', n, 1);
    stft_map_tmp(:, t) = Ps_plot;
end

figure 
imagesc(time_stft-1.2,f,stft_map/(num_channels*num_trials));
ylim([0, 40])
c_bar = colorbar;
caxis([0, 13])
set(gca,'YDir','normal')
title("STFT")
xlabel('Time (s)')
ylabel('Frequency (Hz)')
ylabel(c_bar,'Power (dB)')

figure
imagesc(time_stft-1.2,f(n/2+2:end),stft_map_tmp(n/2+2:end, :)/(num_channels*num_trials));
ylim([2, 40])
c_bar = colorbar;
caxis([0, 13])
set(gca,'YDir','normal')
title("STFT - Denoised (no Pink noise)")
xlabel('Time (s)')
ylabel('Frequency (Hz)')
ylabel(c_bar,'Power (dB)')
% Welch Method
window_length = 80;
num_overlap_samples = 60;
pxx_mean = 0;
for ch_no = 1:num_channels
    lfp_data = chan(ch_no).lfp;
    lfp_data = zscore(lfp_data);
    for trial_no = 1:num_trials
        trial_data = lfp_data(:, trial_no);
        tmp = buffer(trial_data, window_length, num_overlap_samples);
        [pxx,f] = pwelch(tmp, 40, 20, fs, fs);
        pxx_mean = pxx_mean+pxx;
    end
end

pxx_clean = [];
for t = 1:size(pxx_mean, 2)
    Ps = pxx_mean(:, t);
    n = length(Ps);
    Ps_plot = removePinkNoise(Ps, f', n, 2);
    pxx_clean(:, t) = Ps_plot;
end

figure
imagesc(linspace(-1.2, 2, size(tmp, 2)),f, pxx_mean/(num_channels*num_trials))
c_bar = colorbar;
set(gca,'YDir','normal')
title("Power Spectrum over Time - Welch")
xlabel('Time (s)')
ylabel('Frequency (Hz)')
ylabel(c_bar,'Power')
caxis([-0.05, 0.1])
ylim([0, 40])

figure
imagesc(linspace(-1.2, 2, size(tmp, 2)),f, pxx_clean/(num_channels*num_trials))
c_bar = colorbar;
set(gca,'YDir','normal')
title("Power Spectrum over Time - Welch - Denoised (no Pink noise)")
xlabel('Time (s)')
ylabel('Frequency (Hz)')
ylabel(c_bar,'Power')
caxis([-0.05, 0.1])
ylim([0, 40])
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Phase propagation (Traveling waves)
% part a - Bandpass Filter over dominant frequency
clc
close all
dominant_freq = 12.5;
[b, a] = butter(2, [dominant_freq-0.5 dominant_freq+0.5] / (fs * 0.5), 'bandpass');
freqz(b,a,fs,fs)
ax = findall(gcf, 'Type', 'axes');
set(ax, 'XLim', [10 20]);
xline(ax(1), dominant_freq, '--k');
xline(ax(2), dominant_freq, '--k');
title('Frequency Response of the Filter')
for ch_no = 1:num_channels
    lfp_data = chan(ch_no).lfp;
    lfp_data = zscore(lfp_data);
    filtered_data = transpose(filtfilt(b, a, transpose(lfp_data)));
    chan(ch_no).filtered_lfp = filtered_data;
end
%% part b - calculating instantaneous pahse of filtered signals
for ch_no = 1:num_channels
    filtered_lfp_data = chan(ch_no).filtered_lfp;
    instantaneous_phase = cos(angle(hilbert(filtered_lfp_data)));
    chan(ch_no).phase = instantaneous_phase;
end
%% part c - showing the traveling wave
clc
close all

times_plot = 1/fs:1/fs:3.2+1/fs;
times_plot = floor(times_plot*fs);

% making the mat to be shown in the demo
angle_mat_2_show = zeros(size(ChannelPosition, 1), size(ChannelPosition, 2), length(times_plot))*nan;
data_mat_2_show = zeros(size(ChannelPosition, 1), size(ChannelPosition, 2), length(times_plot))*nan;
time_counter = 1;
for t = times_plot
    for i = 1:size(ChannelPosition, 1)
        for j = 1:size(ChannelPosition, 2)
            ch_no = ChannelPosition(i, j);
            if ~isnan(ch_no)
                angle_2_show = chan(ch_no).phase(:, trial_no);
                angle_2_show = angle_2_show(t);
                angle_mat_2_show(i, j, time_counter) = angle_2_show;
                
                data_2_show = chan(ch_no).filtered_lfp(:, trial_no);
                data_2_show = data_2_show(t);
                data_mat_2_show(i, j, time_counter) = angle_2_show;
            end
        end
    end
    time_counter = time_counter+1;
end

focus_chs = [11, 16, 21, 26, 31, 36];

% showing the demo
time_counter = 100;
for t = times_plot(time_counter:end-time_counter)
    time_2_plot = t/fs-1/fs-1.2;
    subplot(2,1,1)
    indexes = time_counter-50:time_counter+50;
    times_2_plot = indexes/fs-1/fs-1.2;
    for ch_no = 1:48
        [row, col] = find(ChannelPosition==ch_no);
        frame_data_2_plot = angle_mat_2_show(row, col, indexes);
        frame_data_2_plot = reshape(frame_data_2_plot, [], length(indexes));
        plot_color = [0.9 0.9 0];
        plot_line_width = 1;
        plot(times_2_plot, frame_data_2_plot, 'color', plot_color, 'LineWidth', plot_line_width)
        hold on
    end
    
    ch_counter = 1;
    for ch_no = focus_chs
        [row, col] = find(ChannelPosition==ch_no);
        frame_data_2_plot = angle_mat_2_show(row, col, indexes);
        frame_data_2_plot = reshape(frame_data_2_plot, [], length(indexes));
        plot_color = [0.8, 0.8, 0.8]*1/ch_counter;
        plot_line_width = 2;
        ch_counter = ch_counter+1;
        plot(times_2_plot, frame_data_2_plot, 'color', plot_color, 'LineWidth', plot_line_width)
    end
            
    xline(time_2_plot, '--r');
    hold off
    ylim([-1.5 , 1.5])
    xlim([times_2_plot(1), times_2_plot(end)])
    xlabel('Time (s)')
    
    
    subplot(2,1,2)
    frame_angle_2_plot = angle_mat_2_show(:, :, time_counter);
    img = imagesc(frame_angle_2_plot);
    hold on
    
    ch_counter = 1;
    for ch_no = focus_chs
        [row, col] = find(ChannelPosition==ch_no);
        plot_color = [0.8, 0.8, 0.8]*1/ch_counter;
        ch_counter = ch_counter+1;
        scatter(col, row, 40, plot_color, 'filled')
    end
    title("Time = "+num2str(time_2_plot))
    hold off
    pause(0.2)
    
    time_counter = time_counter+1;
end
%% Functions

function Ps_plot = removePinkNoise(Ps, f, n, bool)
    if bool==1
        pink_noise = 1./f(n/2+2:end);
        [~,~,spectrum_regressed] = regress(Ps(n/2+2:end), pink_noise');
        pink_spectrum = Ps(n/2+2:end)-spectrum_regressed;
        Ps_plot = Ps;
        Ps_plot(n/2+2:end) = Ps(n/2+2:end)-pink_spectrum;
    else
        pink_noise = 1./f;
        if pink_noise(1)==Inf, pink_noise(1)=pink_noise(2); end
        [~,~,spectrum_regressed] = regress(Ps, pink_noise');
        pink_spectrum = Ps-spectrum_regressed;
        Ps_plot = Ps;
        Ps_plot = Ps-pink_spectrum;
    end
end