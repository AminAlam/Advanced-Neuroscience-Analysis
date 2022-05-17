function [states, transitions] = update_state_and_transition_TD(states_r, states,...
                                                        transitions, direction_nos, learning_rate,...
                                                        discount_factor, target_value, cat_value,...
                                                        softmax_func, agent_locs)

    lambda = 0.99;                                                
    for i = size(agent_locs, 1):-1:2
        agent_loc_past = agent_locs(i-1, :);
        agent_loc = agent_locs(i, :);
        direction_no = direction_nos(i);
        transition = cell2mat(transitions(agent_loc_past(1,1), agent_loc_past(1,2)));
        probs = softmax_func(transition);
        % updating state values
        delta = states_r(agent_loc(1,1), agent_loc(1,2)) + ...
                discount_factor*states(agent_loc(1,1), agent_loc(1,2)) - ...
                states(agent_loc_past(1,1), agent_loc_past(1,2));
        delta = delta*lambda^(size(agent_locs, 1)-i+1);
        states(agent_loc_past(1,1), agent_loc_past(1,2)) = states(agent_loc_past(1,1), agent_loc_past(1,2)) + ...
                                                            learning_rate*delta*probs(1, direction_no);

        if states(agent_loc_past(1,1), agent_loc_past(1,2)) >  target_value/2
            states(agent_loc_past(1,1), agent_loc_past(1,2)) = target_value - 1;
        elseif states(agent_loc_past(1,1), agent_loc_past(1,2)) <  cat_value/2
            states(agent_loc_past(1,1), agent_loc_past(1,2)) = cat_value + 1;
        end

        % updating transition values
        for j = 1:length(transition)
            if j == direction_no
                transition(1, j) = transition(1, j) + (1-probs(1, j)) * delta;
            else
                transition(1, j) = transition(1, j) - probs(1, j) * delta;  
            end
        end
        transitions(agent_loc_past(1,1), agent_loc_past(1,2)) = {transition};
    end
end