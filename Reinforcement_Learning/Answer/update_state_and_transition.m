function [states, transitions] = update_state_and_transition(states_r, states, transitions, direction_no, learning_rate, agent_loc, agent_loc_past, target_value, cat_value)
    transition = cell2mat(transitions(agent_loc_past(1,1), agent_loc_past(1,2)));
    % updating state values
    delta = states_r(agent_loc_past(1,1), agent_loc(1,2)) + ...
            states(agent_loc(1,1), agent_loc(1,2)) - ...
            states(agent_loc_past(1,1), agent_loc(1,2));
        
    states(agent_loc_past(1,1), agent_loc_past(1,2)) = states(agent_loc_past(1,1), agent_loc_past(1,2)) + ...
                                                       learning_rate*states(agent_loc(1,1), agent_loc(1,2));

    if states(agent_loc_past(1,1), agent_loc_past(1,2)) >  target_value/2
        states(agent_loc_past(1,1), agent_loc_past(1,2)) = target_value - 1;
    elseif states(agent_loc_past(1,1), agent_loc_past(1,2)) <  cat_value/2
        states(agent_loc_past(1,1), agent_loc_past(1,2)) = cat_value + 1;
    end

    % updating transition values
    transition(1, direction_no) = transition(1, direction_no) + learning_rate*delta;
    transitions(agent_loc_past(1,1), agent_loc_past(1,2)) = {transition};
end