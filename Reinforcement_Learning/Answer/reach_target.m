function [states, transitions] = reach_target(states, transitions, direction_no, learning_rate, agent_loc, agent_loc_past, target_value, cat_value)
    % updating state values
    states(agent_loc_past(1,1), agent_loc_past(1,2)) = states(agent_loc_past(1,1), agent_loc_past(1,2)) + ...
                                             learning_rate*states(agent_loc(1,1), agent_loc(1,2));
                                         
    if states(agent_loc_past(1,1), agent_loc_past(1,2)) >  target_value/2
        states(agent_loc_past(1,1), agent_loc_past(1,2)) = target_value - 1;
    elseif states(agent_loc_past(1,1), agent_loc_past(1,2)) <  cat_value/2
        states(agent_loc_past(1,1), agent_loc_past(1,2)) = cat_value + 1;
    end
    % updating transition values
    transition = cell2mat(transitions(agent_loc_past(1,1), agent_loc_past(1,2)));
    coeff = states(agent_loc(1,1), agent_loc(1,2)) - states(agent_loc_past(1,1), agent_loc_past(1,2));
    transition(1, direction_no) = transition(1, direction_no) + coeff*learning_rate;
    transitions(agent_loc_past(1,1), agent_loc_past(1,2)) = {transition};
end