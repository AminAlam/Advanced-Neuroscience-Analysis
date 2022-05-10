clc
clear
close all

% loading images
rat_img = imread('assets/rat.png');
cat_img = imread('assets/cat.png');
target_img = imread('assets/target.png');

% configs
map_size = [15, 15];
cat_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
cat_loc = [5, 8];
target_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
target_loc = [9, 11];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_trials = 100;
learning_rate = 0.01;
num_directions = 4;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

% initializing the movement direction probabilities for all the states
states = zeros(map_size);
states(target_loc(1,1), target_loc(1,2)) = 10;
states(cat_loc(1,1), cat_loc(1,2)) = -10;

for i = 1:map_size(1,1)
    for j = 1:map_size(1,2)
        transitions{i,j} = zeros(1,4);
    end
end


softmax_func = @(x) exp(x)/sum(exp(x));
figure('units','normalized','outerposition',[0 0 1 1])

for trial_no = 1:num_trials
    reach_target_bool = 0;
    reach_cat_bool = 0;
    agent_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
    step_no = 1;
    agent_locs = [];
    
    while ~reach_target_bool && ~reach_cat_bool
        agent_loc_past = agent_loc;

        % deciding which direction to choose
        directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))));
        if length(unique(directions_probs)) == 1
           direction_no = ceil(num_directions*rand(1,1));
        else
           [~, direction_no] = find(directions_probs == max(directions_probs)); 
        end
        
        direction = direction_map{direction_no};
        check_mat = map_size - (agent_loc + direction);
        if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < 15 && check_mat(1,2) < 15
           agent_loc = agent_loc + direction;
        else
            while 1
                i = ceil(rand(1,1)*num_directions);
                direction = direction_map{i};
                check_mat = map_size - (agent_loc + direction);
                if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < 15 && check_mat(1,2) < 15
                    agent_loc = agent_loc + direction;
                    break
                end
            end
        end

        % checking location
        if agent_loc == target_loc
            reach_target_bool = 1;
        end
        if agent_loc == cat_loc
            reach_cat_bool = 1;
        end
        agent_locs = [agent_locs; agent_loc];
        [states, transitions] = reach_target(states, transitions, direction_no, learning_rate, agent_loc, agent_loc_past);
        
        subplot(2,3,[1,2,4,5])
        plot_map(agent_loc, agent_locs, target_loc, cat_loc, map_size, rat_img, cat_img, target_img)
        title("Trial "+num2str(trial_no)+" | "+"Step "+num2str(step_no))
        subplot(2,3,3)
        imagesc(states);
        set(gca,'YDir','normal')
        colormap hot
        colorbar
        subplot(2,3,6)
        [fx,fy] = gradient(states);
        quiver(fx,fy)
        step_no = step_no + 1;
        pause(0.05)
        
    end
end