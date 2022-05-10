function [states, transitions] = reach_target(states, transitions, direction_no, learning_rate, agent_loc, agent_loc_past)
    % updating state values
    states(agent_loc_past(1,1), agent_loc_past(1,2)) = states(agent_loc_past(1,1), agent_loc_past(1,2)) + ...
                                             learning_rate*states(agent_loc(1,1), agent_loc(1,2));
    % updating transition values
    transition = cell2mat(transitions(agent_loc_past(1,1), agent_loc_past(1,2)));
    coeff = states(agent_loc(1,1), agent_loc(1,2)) - states(agent_loc_past(1,1), agent_loc_past(1,2));
    transition(1, direction_no) = transition(1, direction_no) + coeff*learning_rate;
    transitions(agent_loc_past(1,1), agent_loc_past(1,2)) = {transition};
end