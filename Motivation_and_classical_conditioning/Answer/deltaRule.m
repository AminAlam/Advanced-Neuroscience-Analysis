function w = deltaRule(w0,num_trials,espsilon,u,r)
    w = zeros(size(w0,1),num_trials);
    w(:,1) = w0;
    for trial_no = 2:num_trials
        v =  w(:,trial_no-1)' * u(:,trial_no-1);
        delta = r(trial_no-1) - v;
        w(:,trial_no) = w(:,trial_no-1) + espsilon*delta*u(:,trial_no-1);
    end
end