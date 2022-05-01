function [w, cov_mat] = kalamFilterInhanced(u, r, cov0, w0, tau, noise_p, num_trials)
    w = zeros(size(w0,1),num_trials);
    w(:,1) = w0;
    cov_mat = zeros(2, 2, num_trials);
    cov_mat(:,:, 1) = cov0;
    for trial_no = 2:num_trials
        cov_mat_past = cov_mat(:,:,trial_no-1) + noise_p;
        coeff = cov_mat_past*u(:, trial_no-1)/(u(:, trial_no-1)'*cov_mat_past*u(:, trial_no-1) + tau^2);
        cov_mat(:,:,trial_no) = cov_mat_past - coeff*u(:,trial_no-1)'*cov_mat_past;
        w(:,trial_no) = w(:,trial_no-1) + coeff*(r(trial_no-1) - w(:,trial_no-1)' * u(:,trial_no-1));
    end
end