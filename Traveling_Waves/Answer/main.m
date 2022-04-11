clc
clear
close all
load("data/ArrayData.mat")
load("data/CleanTrials.mat")
fs = 200;
save_figures = 0;
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
normalize_constant = 10*log10((num_channels*num_trials)^2/n);

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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/a_1",'-dpng','-r0')
end

figure
plot(f(n/2+2:end), pink_spectrum, '--r', 'LineWidth', 2)
hold on
plot(f(n/2+2:end), Ps_plot(n/2+2:end)-normalize_constant, 'k', 'LineWidth', 2)
title('Averaged power spectrum of all trials of all channels (normalized)')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
xlim([1, 100])
legend('Estimated Pink Noise')
grid on

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/a_2",'-dpng','-r0')
end

% removing pink noise
figure
hold on
spectrum_clean = Ps_plot(n/2+2:end) - pink_spectrum;
plot(f(n/2+2:end), Ps_plot(n/2+2:end)- normalize_constant, 'r', 'LineWidth', 2) 
plot(f(n/2+2:end), spectrum_clean - normalize_constant, 'k', 'LineWidth', 2) 
legend('Original', 'Denoised (No Pink Noise)')
title('Averaged power spectrum of all trials of all channels (normalized)')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
xlim([1, 70])
ylim([-10, 40])
grid on

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/a_3",'-dpng','-r0')
end
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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/b_1",'-dpng','-r0')
end

figure
plt = imagesc(dominant_freq_mat);
set(plt,'AlphaData', ~isnan(dominant_freq_mat))
colormap jet
c_bar = colorbar;
caxis([0, 13])
title('Dominant Frequencies')
ylabel(c_bar,'Frequency (Hz)')

textStrings = num2str(dominant_freq_mat(:), '%0.2f');
textStrings = strtrim(cellstr(textStrings));  
[x, y] = meshgrid(1:10, 1:5);  
hStrings = text(x(:), y(:), textStrings(:), ...  
                'HorizontalAlignment', 'center');
midValue = mean(get(gca, 'CLim'));  
textColors = repmat(dominant_freq_mat(:) > midValue, 1, 3);  
set(hStrings, {'Color'}, num2cell(textColors, 2));

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/b_2",'-dpng','-r0')
end
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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/c_1",'-dpng','-r0')
end

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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/c_2",'-dpng','-r0')
end

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
        pxx_mean = pxx_mean + pxx;
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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/c_3",'-dpng','-r0')
end

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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/c_4",'-dpng','-r0')
end

%% Wavelet

wt_map = 0;
for ch_no = 1:num_channels
    lfp_data = chan(ch_no).lfp;
    lfp_data = zscore(lfp_data);
    for trial_no = 1:num_trials
        trial_data = lfp_data(:, trial_no);
        [wt,f,coi] = cwt(trial_data,fs);
        wt_map = wt_map + abs(wt);
    end
end

figure
surface(Time, f, wt_map/(num_channels*num_trials));
axis tight
shading flat
c_bar = colorbar;
ylim([1.2, 40])
set(gca,'YDir','normal')
title("Power Spectrum over Time - Wavelet")
xlabel('Time (s)')
ylabel('Frequency (Hz)')
ylabel(c_bar,'Power (dB)')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/c_5",'-dpng','-r0')
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Phase propagation (Traveling waves)
% part a - Bandpass Filter over dominant frequency
clc
close all
dominant_freq = 12.5;
[b, a] = butter(2, [dominant_freq-1 dominant_freq+1] / (fs * 0.5), 'bandpass');
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

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/2_a",'-dpng','-r0')
end
%% part b - calculating instantaneous pahse of filtered signals
for ch_no = 1:num_channels
    filtered_lfp_data = chan(ch_no).filtered_lfp;
    instantaneous_phase = angle(hilbert(filtered_lfp_data));
    instantaneous_cos_phase = cos(angle(hilbert(filtered_lfp_data)));
    chan(ch_no).phase = instantaneous_phase;
    chan(ch_no).cos_phase = instantaneous_cos_phase;
end
%% part c - showing the traveling wave
clc
close all
times_plot = 1/fs:1/fs:3.2+1/fs;
times_plot = floor(times_plot*fs);
trial_no = 259;

video_boolian = 1;

% making the mat to be shown in the demo
angle_mat_2_show = zeros(size(ChannelPosition, 1), size(ChannelPosition, 2), length(times_plot))*nan;
angle_cos_mat_2_show = zeros(size(ChannelPosition, 1), size(ChannelPosition, 2), length(times_plot))*nan;
data_mat_2_show = zeros(size(ChannelPosition, 1), size(ChannelPosition, 2), length(times_plot))*nan;
time_counter = 1;
for t = times_plot
    for i = 1:size(ChannelPosition, 1)
        for j = 1:size(ChannelPosition, 2)
            ch_no = ChannelPosition(i, j);
            if ~isnan(ch_no)
                angle_2_show = chan(ch_no).cos_phase(:, trial_no);
                angle_2_show = angle_2_show(time_counter);
                angle_mat_2_show(i, j, time_counter) = angle_2_show;
                
                angle_cos_2_show = chan(ch_no).cos_phase(:, trial_no);
                angle_cos_2_show = angle_cos_2_show(time_counter);
                angle_cos_mat_2_show(i, j, time_counter) = angle_cos_2_show;
                
                data_2_show = chan(ch_no).filtered_lfp(:, trial_no);
                data_2_show = data_2_show(time_counter);
                data_mat_2_show(i, j, time_counter) = angle_2_show;
            end
        end
    end
    time_counter = time_counter+1;
end

focus_chs = [11, 16, 21, 26, 31, 36];

% showing the demo
fig = figure('units','normalized','outerposition',[0 0 1 1]);
if video_boolian
    writerObj = VideoWriter("videos/Demo_Trial"+num2str(trial_no));
    writerObj.FrameRate = 25;
    open(writerObj);
end
time_counter = 101;
phi_0 = 0;
for t = times_plot(time_counter:end-time_counter)
    time_2_plot = t/fs-1/fs-1.2;
    subplot(2,2,[1, 2])
    indexes = time_counter-100:time_counter+100;
    times_2_plot = indexes/fs-1/fs-1.2;
    for ch_no = 1:48
        [row, col] = find(ChannelPosition==ch_no);
        frame_data_2_plot = angle_mat_2_show(row, col, indexes);
        frame_data_2_plot = reshape(frame_data_2_plot, [], length(indexes));
        plot_color = [0.9 0.9 0.9];
        plot_line_width = 1;
        plot(times_2_plot, frame_data_2_plot, 'color', plot_color, 'LineWidth', plot_line_width)
        hold on
    end
    
    ch_counter = 1;
    plot_color = [0, 0, 0];
    for ch_no = focus_chs
        [row, col] = find(ChannelPosition==ch_no);
        frame_data_2_plot = angle_mat_2_show(row, col, indexes);
        frame_data_2_plot = reshape(frame_data_2_plot, [], length(indexes));
        plot_color = plot_color + 0.1;
        plot_line_width = 2;
        ch_counter = ch_counter+1;
        plot(times_2_plot, frame_data_2_plot, 'color', plot_color, 'LineWidth', plot_line_width)
    end
    title("Trial No."+num2str(trial_no))
    xline(time_2_plot, '--r');
    hold off
    ylim([-1.5 , 1.5])
    xlim([times_2_plot(1), times_2_plot(end)])
    xlabel('Time (s)')
    
    
    subplot(2,2,3)
    frame_angle_2_plot = angle_mat_2_show(:, :, time_counter);
    img = imagesc(frame_angle_2_plot);
    colormap hot
    set(img,'AlphaData', ~isnan(frame_angle_2_plot))
    hold on
    ch_counter = 1;
    plot_color = [0, 0, 0];
    for ch_no = focus_chs
        [row, col] = find(ChannelPosition==ch_no);
        plot_color = plot_color + 0.1;
        ch_counter = ch_counter+1;
        scatter(col, row, 80, plot_color, 'filled')
    end
    title("Time = "+num2str(time_2_plot))
    hold off    
    colorbar
    caxis([-1, 1])
    
    subplot(2,2,4)
    [u,v] = gradient(frame_angle_2_plot,1,1);
    [y,x] = ndgrid(1:5,1:10);
    quiver(x,y,u,v)
    hold on
    
    u_mean = nanmean(u, 'all');
    v_mean = nanmean(v, 'all');
    u_mean = 10*u_mean;
    v_mean = 10*v_mean;
    
    u = u/400e-6;
    v = v/400e-6;
    
    plot([5, 5+u_mean], [3, 3+v_mean], 'k->', 'LineWidth', 2)
    
    ch_counter = 1;
    plot_color = [0, 0, 0];
    for ch_no = focus_chs
        [row, col] = find(ChannelPosition==ch_no);
        plot_color = plot_color + 0.1;
        ch_counter = ch_counter+1;
        scatter(col, row, 80, plot_color, 'filled')
    end
    
    ylim([0.5, 5.5])
    xlim([0.5, 10.5])
    hold off
    pgd = pgdCalculator(u, v);
    d_phi = frame_angle_2_plot - phi_0;
    speed = 100*speedCalculator(u, v, d_phi, 1/fs);
    phi_0 = frame_angle_2_plot;
    title("PGD = "+num2str(pgd)+" | Speed = "+num2str(speed))
    legend('Gradient Vectors', 'Avg Gradient Vector')
    
    axis ij

    time_counter = time_counter+1;
    if video_boolian
        frame = getframe(fig);
        for frame_index = 1:4
            writeVideo(writerObj,frame);
        end
    else
        pause(0.1)
    end
end
if video_boolian
    close(writerObj)
end

%% calculating parameters for all trials

clc
close all
times_plot = 1/fs:1/fs:3.2+1/fs;
times_plot = floor(times_plot*fs);
num_trials = size(chan(1).lfp, 2);
angle_mat_2_show = zeros(size(ChannelPosition, 1), size(ChannelPosition, 2), length(times_plot))*nan;
pdg_mat = zeros(num_trials, length(times_plot));
speed_mat = zeros(num_trials, length(times_plot));
gradient_direction_mat_mean = zeros(num_trials, length(times_plot));
gradient_direction_mat_all = zeros(num_trials, length(times_plot), 5, 5)*nan;

for trial_no = 1:size(chan(1).lfp, 2)
    time_counter = 1;
    phi_0 = 0;
    for t = times_plot
        for i = 1:size(ChannelPosition, 1)
            for j = 1:size(ChannelPosition, 2)
                ch_no = ChannelPosition(i, j);
                if ~isnan(ch_no)
                    angle_2_show = chan(ch_no).phase(:, trial_no);
                    angle_2_show = angle_2_show(time_counter);
                    angle_mat_2_show(i, j, time_counter) = angle_2_show;
                end
            end
        end
        [u, v] = gradient(angle_mat_2_show(:,:,time_counter),1,1);
        u = u/400e-6;
        v = v/400e-6;
        pdg_mat(trial_no, time_counter) = pgdCalculator(u, v);
        d_phi = angle_mat_2_show(:,:,time_counter) - phi_0;
        speed_mat(trial_no, time_counter) = speedCalculator(u, v, d_phi, 1/fs);
        phi_0 = angle_mat_2_show(:,:,time_counter);
        for row_grad = 1:size(u, 1)
            for col_grad = 1:size(u, 2)
                ch_no = ChannelPosition(row_grad, col_grad);
                if ~isnan(ch_no)
                    a = [u(row_grad, col_grad), v(row_grad, col_grad), 0];
                    b = [1, 0, 0];
                    ThetaInDegrees = atan2(norm(cross(a,b)),dot(a,b))*180/pi;
                    gradient_direction_mat_all(trial_no, time_counter, row_grad, col_grad) = ThetaInDegrees;
                end
            end
        end
        a = [nanmean(u, 'all'), nanmean(v, 'all'), 0];
        b = [1, 0, 0];
        thetaInDegrees = atan2(norm(cross(a,b)),dot(a,b))*180/pi;
        gradient_direction_mat_mean(trial_no, time_counter) = thetaInDegrees;
        time_counter = time_counter+1;
    end
end
%% plotting hitogram of PGDs and gradient directions
figure
histogram(gradient_direction_mat_mean(:,1:241), 200,'Normalization', 'pdf')
hold on
histogram(gradient_direction_mat_mean(:,241:end), 200,'Normalization', 'pdf')
title('PDF of Direction of Wave Propagation')
legend('Before Onset', 'After Onset')
xlabel('Propagation Direction (degree)')
ylabel('Probability Density')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/2_e_1",'-dpng','-r0')
end

figure
gradient_direction_mat_all_tmp = gradient_direction_mat_all(:, 1:241, :, :);
gradient_direction_mat_all_tmp = gradient_direction_mat_all_tmp(~isnan(gradient_direction_mat_all_tmp));
histogram(gradient_direction_mat_all_tmp, 200,'Normalization', 'pdf')
hold on
gradient_direction_mat_all_tmp = gradient_direction_mat_all(:, 241:end, : ,:);
gradient_direction_mat_all_tmp = gradient_direction_mat_all_tmp(~isnan(gradient_direction_mat_all_tmp));
histogram(gradient_direction_mat_all_tmp, 200,'Normalization', 'pdf')
title('PDF of Direction of Gradient of all channels')
legend('Before Onset', 'After Onset')
xlabel('Gradient Direction (degree)')
ylabel('Probability Density')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/2_e_2",'-dpng','-r0')
end

figure
histogram(speed_mat(:,1:241)*100, 200,'Normalization', 'pdf')
hold on
histogram(speed_mat(:,241:end)*100, 200,'Normalization', 'pdf')
title('PDF of Speed of Wave Propagation')
legend('Before Onset', 'After Onset')
xlabel('Speed (cm/s)')
ylabel('Probability Density')
xlim([0, 50])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/2_e_3",'-dpng','-r0')
end
%% repeating last part just for times with pdg > 0.5

[rows, cols] = find(pdg_mat<=0.5);

gradient_direction_mat_mean_backup = gradient_direction_mat_mean;
gradient_direction_mat_all_backup = gradient_direction_mat_all;
speed_mat_backup = speed_mat;

for i = 1:length(rows)
    row = rows(i);
    col = cols(i);
    gradient_direction_mat_mean_backup(row,col) = nan;
    gradient_direction_mat_all_backup(row,col, :, :) = nan;
    speed_mat_backup(row,col) = nan;
end

figure
histogram(gradient_direction_mat_mean_backup, 200,'Normalization', 'pdf')
title('PDF of Direction of Wave Propagation')
xlabel('Propagation Direction (degree)')
ylabel('Probability Density')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/3_2_e_1",'-dpng','-r0')
end

figure
gradient_direction_mat_all_tmp = gradient_direction_mat_all_backup;
gradient_direction_mat_all_tmp = gradient_direction_mat_all_tmp(~isnan(gradient_direction_mat_all_tmp));
histogram(gradient_direction_mat_all_tmp, 200,'Normalization', 'pdf')
title('PDF of Direction of Gradient of all channels')
xlabel('Gradient Direction (degree)')
ylabel('Probability Density')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/3_2_e_2",'-dpng','-r0')
end

figure
histogram(speed_mat_backup*100, 200,'Normalization', 'pdf')
title('PDF of Speed of Wave Propagation')
xlabel('Speed (cm/s)')
ylabel('Probability Density')
xlim([0, 50])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/3_2_e_3",'-dpng','-r0')
end

%% part f - validating the observations
clc
figure
h = histogram(gradient_direction_mat_mean_backup, 200,'Normalization', 'pdf');
values = h.Values;
bins = h.BinEdges(1:end-1)-90;
values = [values(end/2+1:end) values(1:end/2)];
plot(bins,values, 'LineWidth', 2)
xlabel('Propagation Direction (degree)')
ylabel('Probability Density')
title('PDF of Direction of Wave Propagation (wraped between -90 and 90)')

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/2_f",'-dpng','-r0')
end

[h,p,~,~] = ttest(bins)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2D Fourier method
clc
close all

times_plot = 1/fs+51/fs:1/fs:3.2+1/fs-51/fs;
times_plot = floor(times_plot*fs);

ch_list = [2, 6, 11, 16, 21, 26, 31, 36, 41, 46];

stacked_mat = zeros(length(ch_list), size(chan(1).lfp, 1), size(chan(1).lfp, 2)) ;

ch_counter = 1;
for ch_no = ch_list
    stacked_mat(ch_counter, :, :) = chan(ch_no).filtered_lfp;
    ch_counter = ch_counter+1;
end

max_values = [];
for t = times_plot
    max_values_tmp = [];
    for trial_no = 1:size(chan(1).filtered_lfp, 2)
        fft_stacked_mat(:, :, trial_no) = fft2(stacked_mat(:, t-50:t+50, trial_no));
    

        freqs_x = (-1*L/2+1:L/2)/L*fs;
        [~, col] = find(abs(freqs_x-dominant_freq) == min(abs(freqs_x-dominant_freq)));
        fft_stacked_mat = mean(abs(fft_stacked_mat), 3);
        fft_slice = fft_stacked_mat(:, col);
        L = size(fft_stacked_mat, 2);
    %     imagesc(freqs_x, 1:10, fftshift(fft_stacked_mat))
    %     xlim([-20, 20])
        max_values_tmp = [max_values_tmp, max(fft_slice)];
    end
    max_values = [max_values; max_values_tmp]; 
end
%%
% plot(max_values)

%% Functions

function pgd = pgdCalculator(fx, fy)
    num = norm(nanmean(fx, 'all'), nanmean(fy, 'all'));
    den = nanmean(sqrt(fx.^2+fy.^2), 'all');
    pgd = num/den;
end

function speed = speedCalculator(fx, fy, dphi, dt)
    num = abs(nanmean(dphi/dt, 'all'));
    den = nanmean(sqrt(fx.^2+fy.^2), 'all');
    speed = num/den;
end

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