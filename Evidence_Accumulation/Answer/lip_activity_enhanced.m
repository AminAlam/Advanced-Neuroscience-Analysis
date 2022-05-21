function [LIP_spikes, MT, times] = lip_activity_enhanced(LIP_weights, stimuli_values, MT1_dist, MT2_dist, num_iters)
    dt = 0.001;
    N = [0; 0];
    LIP_spikes = [];
    t = 0;
    MT = [];
    times = [];
    for iter_no = 1:num_iters
        stimuli = stimuli_values(randi(length(stimuli_values)));
        times = [times, t];
        t = t + dt;
        firing_probs = [pdf(MT1_dist, stimuli); pdf(MT2_dist, stimuli)];
        dN = rand(2, 1) < firing_probs;
        N = N + dN;
        MT = [MT, dN];
        p_LIP = N'*LIP_weights;
        LIP_event = rand(2,1) < p_LIP';
        LIP_spikes = [LIP_spikes, LIP_event];
    end
end