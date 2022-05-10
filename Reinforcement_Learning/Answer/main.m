clc
clear
close all

% configs
map_size = [15, 15];
cat_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
target_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
 
map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = 1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 2;

num_trials = 10;
increament_prob = 0.05;
num_directions = 4;
direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

% initializing the movement direction probabilities for all the states
for i = 1:map_size(1,1)
    for j = 1:map_size(1,2)
        states{i,j} = ones(1,4)*0.25;
    end
end
%%
figure
for trial_no = 1:num_trials
    reach_target_bool = 0;
    reach_cat_bool = 0;
    agent_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
    step_no = 1;

    while ~reach_target_bool && ~reach_cat_bool
        agent_loc_past = agent_loc;

        % deciding which direction to choose
        directions_probs = cell2mat(states(agent_loc(1,1), agent_loc(1,2)));
        if length(unique(directions_probs)) == 1
           direction_no = ceil(num_directions*rand(1,1));
        else
           [~, direction_no] = find(directions_probs == max(directions_probs)); 
        end
        direction = direction_map{direction_no};
        check_mat = map_size - (agent_loc + direction);
        if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < 15 && check_mat(1,2) < 15
           agent_loc = agent_loc + direction;
        end
        
        % checking location
        if agent_loc == target_loc
            reach_target_bool = 1;
            states = reach_target(states, increament_prob, direction_no, num_directions, agent_loc_past);
        end
        if agent_loc == cat_loc
            reach_cat_bool = 1;
            states = reach_cat(states, increament_prob, direction_no, num_directions, agent_loc_past);
        end
        
        mat_map_tmp = map_mat_main;
        mat_map_tmp(agent_loc(1,1), agent_loc(1,2)) = 3;
        imagesc(mat_map_tmp)
        pause(0.2)
        step_no = step_no + 1;
    end
end