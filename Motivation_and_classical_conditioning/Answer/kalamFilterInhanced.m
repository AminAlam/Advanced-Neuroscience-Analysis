function [w, cov_mat, B] = kalamFilterInhanced(u, r, cov0, w0, tau, gamma, noise_p, phi_noise, num_trials)
    w = zeros(size(w0,1),num_trials);
    B = zeros(1, num_trials);
    w(:,1) = w0;
    cov0 = eye(size(w0,1))*cov0;
    noise_p = eye(size(w0,1))*noise_p;
    cov_mat = zeros(size(w0,1), size(w0,1), num_trials);
    cov_mat(:,:, 1) = eye(size(w0,1))*cov0;

    for trial_no = 2:num_trials
        cov_mat_past = cov_mat(:,:,trial_no-1) + noise_p;
        
        coeff = cov_mat_past*u(:, trial_no-1)/(u(:, trial_no-1)'*cov_mat_past*u(:, trial_no-1) + tau^2);

        cov_mat(:,:,trial_no) = cov_mat_past - coeff*u(:,trial_no-1)'*cov_mat_past;
        w(:,trial_no) = w(:,trial_no-1) + coeff*(r(trial_no-1) - w(:,trial_no-1)' * u(:,trial_no-1));
        
        
        B(trial_no) = (r(trial_no) - w(:,trial_no)'*u(:, trial_no))^2/(u(:, trial_no)'*cov_mat_past*u(:, trial_no) + tau^2);
        if B(trial_no) < gamma
            c = 0;
        else
            c = 1;
        end
        phi_noise_mat = c*ones(size(w0,1),1)*phi_noise;
        
        cov_mat(:,:,trial_no) = cov_mat(:,:,trial_no) + phi_noise_mat;
    end
end