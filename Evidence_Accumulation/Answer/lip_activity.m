function rt = lip_activity(MT_p_values,LIP_weights,LIP_threshold)
    % Parameters:
    % MT_p_values - a vector with 2 elements, firing probabilities for the
    % excitatory and inhibitory neurons, resp.
    % LIP_weights - a length 2 vector of weighting factors for the evidence
    % from the excitatory (positive) and
    % inhibitory (negative) neurons
    % LIP_threshold - the LIP firing rate that represents the choice threshold criterion % use fixed time scale of 1 ms
    dt = 0.001;
    N = [0 0]; % plus is first, minus is second
    rate = 0.0;
    LIP_event_times = [];
    while rate < LIP_threshold
    t = t + 0.001;
    dN = rand(2) < MT_p_values;
    N = N + dN;
    p_LIP = sum(N.*LIP_weights);
    LIP_event = rand < p_LIP;
    _event_times = [LIP_event_times t];
    % check LIP mean rate for last M spikes rate = M/(t - LIP_event_times(N_LIP - M));
    end
    rt = t;
end