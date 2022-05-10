function states = reach_target(states, increament_prob, agent_loc, agent_loc_past)
    states(agent_loc_past(1,1), agent_loc_past(1,2)) = states(agent_loc_past(1,1), agent_loc_past(1,2)) + ...
                                             increament_prob*states(agent_loc(1,1), agent_loc(1,2));
end