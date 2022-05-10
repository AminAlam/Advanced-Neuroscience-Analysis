function states = reach_cat(states, increament_prob, direction_no, num_directions, agent_loc_past);
    state = cell2mat(states(agent_loc_past(1,1), agent_loc_past(1,2)));
    for i = 1:num_directions
        if i == direction_no
            coeff = -1;
        else
            coeff = 1/(num_directions-1);
        end
        coeff
        state
        state(1, i) = state(1, i) + coeff*increament_prob;
    end
    states(agent_loc_past(1,1), agent_loc_past(1,2)) = {state};
end