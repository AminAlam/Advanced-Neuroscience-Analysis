function states = reach_target(states, increament_prob, direction_no, num_directions, agent_loc_past);
    state = cell2mat(states(agent_loc_past(1,1), agent_loc_past(1,2)));
    for i = 1:num_directions
        if i == direction_no
            coeff = 1;
        else
            coeff = -1/(num_directions-1);
        end
        state(1, direction_no) = state(1, direction_no) + coeff*increament_prob;
    end
    states(agent_loc_past(1,1), agent_loc_past(1,2)) = {state};
end

    