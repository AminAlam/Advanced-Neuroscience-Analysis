clc
clear
close all

% configs
map_size = [15, 15];
cat_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
target_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_trials = 100;
increament_prob = 0.1;
num_directions = 4;
direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

% initializing the movement direction probabilities for all the states
states = zeros(map_size);
states(target_loc(1,1), target_loc(1,2)) = 10;
states(cat_loc(1,1), cat_loc(1,2)) = -10;

softmax_func = @(x) exp(x)/sum(exp(x));
figure
for trial_no = 1:num_trials
    reach_target_bool = 0;
    reach_cat_bool = 0;
    agent_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
    step_no = 1;

    while ~reach_target_bool && ~reach_cat_bool
        agent_loc_past = agent_loc;

        % deciding which direction to choose
        if agent_loc(1,1) == 1
            left_state = 0;
        else
            left_state = states(agent_loc(1,1)-1, agent_loc(1,2));
        end
        
        if  agent_loc(1,1) == map_size(1,1)
            right_state = 0;
        else
            right_state = states(agent_loc(1,1)+1, agent_loc(1,2));
        end
        
        if  agent_loc(1,2) == 1
            down_state = 0;
        else
            down_state = states(agent_loc(1,1), agent_loc(1,2)-1);
        end
        
        if  agent_loc(1,2) == map_size(1,2)
            up_state = 0;
        else
            up_state = states(agent_loc(1,1), agent_loc(1,2)+1);
        end
        
        directions_probs = softmax_func([up_state, down_state, left_state, right_state])
                        
        if length(unique(directions_probs)) == 1
           direction_no = ceil(num_directions*rand(1,1));
        else
           [~, direction_no] = find(directions_probs == max(directions_probs)); 
           direction_no = direction_no(ceil(rand(1,1)*length(direction_no)));
        end
        direction = direction_map{direction_no};
        check_mat = map_size - (agent_loc + direction);
        if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < 15 && check_mat(1,2) < 15
           agent_loc = agent_loc + direction;
        end

        % checking location
        if agent_loc == target_loc
            reach_target_bool = 1;
        end
        if agent_loc == cat_loc
            reach_cat_bool = 1;
        end
        states = reach_target(states, increament_prob, agent_loc, agent_loc_past);
        
        mat_map_tmp = map_mat_main;
        mat_map_tmp(agent_loc(1,1), agent_loc(1,2)) = 3;
        subplot(2,1,1)
        imagesc(mat_map_tmp)
        subplot(2,1,2)
        imagesc(states)
        colormap jet
        colorbar
        step_no = step_no + 1;
        pause(0.05)
    end
end