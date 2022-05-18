clc
clear
close all

save_figures = 0;
video_boolian = 1;
show_plot = 0;
save_freq = 1;

% loading images
rat_img = imread('assets/rat.png');
cat_img = imread('assets/cat.png');
target_img = imread('assets/target.png');

% configs
target_value = 2;
cat_value = -2;
num_trials = 500;
learning_rate_main = 0.1;
forgetting_factor = 1;
discount_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
cat_loc = [12, 13];
target_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
target_loc = [7, 7];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_directions = 4;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

% initializing the movement direction probabilities
states = zeros(map_size);

for i = 1:map_size(1,1)
    for j = 1:map_size(1,2)
        transitions{i,j} = zeros(1,4);
    end
end

% defining the softmax function
softmax_func = @(x) exp(x)/sum(exp(x));

fig = figure('units','normalized','outerposition',[0 0 1 1]);


if video_boolian
    writerObj = VideoWriter("videos/Demo_simpleRL");
    writerObj.FrameRate = 50;
    writerObj.Quality = 20;
    open(writerObj);
end

states_r = zeros(map_size);
states_r(target_loc(1,1), target_loc(1,2)) = target_value;
states_r(cat_loc(1,1), cat_loc(1,2)) = cat_value;
num_steps = zeros(1, num_trials);
for trial_no = 1:num_trials
    % applying the forgeting factor
    states = states*forgetting_factor;

    learning_rate = learning_rate_main/(1+log10(trial_no));
    reach_target_bool = 0;
    reach_cat_bool = 0;
    agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
    agent_loc_past = [1, 1];
    step_no = 1;
    agent_locs = [];
    
    if mod(trial_no-1, save_freq)==0
        show_plot = 1;
    else
        show_plot = 0;
    end
    
    while ~reach_target_bool && ~reach_cat_bool
        

        % deciding which direction to choose
        directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))))
        direction_no = choose_by_prob(directions_probs)
        
        direction = direction_map{direction_no};
        check_mat = map_size - (agent_loc + direction);
        if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
           agent_loc = agent_loc + direction;
        else
            for i = randperm(num_directions)
                direction = direction_map{i};
                check_mat = map_size - (agent_loc + direction);
                if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                    agent_loc = agent_loc + direction;
                    pass_bool = 1;
                    break
                end
            end
            if ~pass_bool
                agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
            end
        end

        % checking location
        
        agent_locs = [agent_locs; agent_loc];
        
        [states, transitions] = update_state_and_transition(states_r, states, transitions, direction_no, learning_rate, discount_factor, agent_loc, agent_loc_past, target_value, cat_value, softmax_func);
        
        if agent_loc == target_loc
            reach_target_bool = 1;
            states(target_loc(1,1), target_loc(1,2)) = target_value;
        end
        if agent_loc == cat_loc
            reach_cat_bool = 1;
            states(cat_loc(1,1), cat_loc(1,2)) = cat_value;
        end

        if show_plot
            subplot(2,3,[1,2,4,5])
            plot_map(agent_loc, agent_locs, target_loc, cat_loc, map_size, rat_img, cat_img, target_img)
            title("Trial "+num2str(trial_no)+" | "+"Step "+num2str(step_no)+" | LR "+num2str(learning_rate))
            subplot(2,3,3)
            imagesc(states);
            set(gca,'YDir','normal')
            colormap bone
            colorbar
            caxis([cat_value, target_value])
            title('States')
            subplot(2,3,6)
            [fx,fy] = gradient(states);
            quiver(fx,fy)
            title('States Gradients')            
        end
        
        step_no = step_no + 1;
        agent_loc_past = agent_loc;
        
        if video_boolian
            frame = getframe(fig);
            for frame_index = 1:2
                writeVideo(writerObj,frame);
            end
        else
            pause(0.01)
        end
    
        if step_no>max_steps
            break
        end
    end
    num_steps(trial_no) = step_no;
    if mod(trial_no-1, save_freq)==0
        if save_figures
            set(gcf,'PaperPositionMode','auto')
            print("images/trial_"+num2str(trial_no),'-dpng','-r0')
        end
    end
    
end
%%
if video_boolian
    close(writerObj)
end
%% Step numbers vs trials

clc
close all

save_figures = 1;

% configs
target_value = 2;
cat_value = -2;
num_trials = 1001;
trial_step = 2;
num_iters = 200;
learning_rate_main = 0.1;
forgetting_factor = 1;
discount_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
cat_loc = [12, 13];
target_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
target_loc = [7, 7];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_directions = 4;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

num_steps_rl = zeros(num_iters, num_trials);

for iter_no = 1:num_iters
    iter_no
    % initializing the movement direction probabilities
    states = zeros(map_size);

    for i = 1:map_size(1,1)
        for j = 1:map_size(1,2)
            transitions{i,j} = zeros(1,4);
        end
    end

    % defining the softmax function
    softmax_func = @(x) exp(x)/sum(exp(x));


    states_r = zeros(map_size);
    states_r(target_loc(1,1), target_loc(1,2)) = target_value;
    states_r(cat_loc(1,1), cat_loc(1,2)) = cat_value;

    for trial_no = 1:trial_step:num_trials
        % applying the forgeting factor
        states = states*forgetting_factor;

        learning_rate = learning_rate_main/(1+log10(trial_no));
        reach_target_bool = 0;
        reach_cat_bool = 0;
        agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
        agent_loc_past = [1, 1];
        step_no = 1;
        agent_locs = [];    

        while ~reach_target_bool && ~reach_cat_bool


            % deciding which direction to choose
            directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))));
            direction_no = choose_by_prob(directions_probs);

            direction = direction_map{direction_no};
            check_mat = map_size - (agent_loc + direction);
            if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
               agent_loc = agent_loc + direction;
            else
                for i = randperm(num_directions)
                    direction = direction_map{i};
                    check_mat = map_size - (agent_loc + direction);
                    if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                        agent_loc = agent_loc + direction;
                        pass_bool = 1;
                        break
                    end
                end
                if ~pass_bool
                    agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                end
            end

            % checking location

            agent_locs = [agent_locs; agent_loc];

            [states, transitions] = update_state_and_transition(states_r, states, transitions, direction_no, learning_rate, discount_factor, agent_loc, agent_loc_past, target_value, cat_value, softmax_func);

            if agent_loc == target_loc
                reach_target_bool = 1;
                states(target_loc(1,1), target_loc(1,2)) = target_value;
            end
            if agent_loc == cat_loc
                reach_cat_bool = 1;
                states(cat_loc(1,1), cat_loc(1,2)) = cat_value;
            end

            step_no = step_no + 1;
            agent_loc_past = agent_loc;

            if step_no>max_steps
                break
            end
        end
        num_steps_rl(iter_no, trial_no) = step_no;
    end
end

plot(1:trial_step:num_trials, mean(num_steps_rl(:, 1:trial_step:num_trials), 1), 'k', 'LineWidth', 1)
xlabel('Trial No')
ylabel('Num of Steps')
title('Num of steps to reach target vs Trials')
xlim([1, num_trials])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/Q02",'-dpng','-r0')
end
%% Effect of learning factor and discount factor
clc
clear
close all

save_figures = 1;

learning_rates = 0.1:0.3:1;
discount_factors = 0.1:0.3:1;

% configs
target_value = 2;
cat_value = -2;
num_trials = 400;
num_iters = 10;
forgetting_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [12, 13];
target_loc = [7, 7];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_directions = 4;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

steps_mat = zeros(length(learning_rates), length(discount_factors));

num_steps = zeros(num_iters, num_trials);
lr_index = 0;

for learning_rate = learning_rates
    learning_rate
    lr_index = lr_index+1;
    df_index = 0;
    for discount_factor = discount_factors
        discount_factor
        df_index = df_index+1;
        for iter_no = 1:num_iters
            % initializing the movement direction probabilities
            states = zeros(map_size);

            for i = 1:map_size(1,1)
                for j = 1:map_size(1,2)
                    transitions{i,j} = zeros(1,4);
                end
            end

            % defining the softmax function
            softmax_func = @(x) exp(x)/sum(exp(x));


            states_r = zeros(map_size);
            states_r(target_loc(1,1), target_loc(1,2)) = target_value;
            states_r(cat_loc(1,1), cat_loc(1,2)) = cat_value;

            for trial_no = 1:num_trials
                % applying the forgeting factor
                states = states*forgetting_factor;

                reach_target_bool = 0;
                reach_cat_bool = 0;
                agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                agent_loc_past = [1, 1];
                step_no = 1;
                agent_locs = [];    

                while ~reach_target_bool && ~reach_cat_bool


                    % deciding which direction to choose
                    directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))));
                    direction_no = choose_by_prob(directions_probs);

                    direction = direction_map{direction_no};
                    check_mat = map_size - (agent_loc + direction);
                    if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                       agent_loc = agent_loc + direction;
                    else
                        for i = randperm(num_directions)
                            direction = direction_map{i};
                            check_mat = map_size - (agent_loc + direction);
                            if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                                agent_loc = agent_loc + direction;
                                pass_bool = 1;
                                break
                            end
                        end
                        if ~pass_bool
                            agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                        end
                    end

                    % checking location

                    agent_locs = [agent_locs; agent_loc];

                    [states, transitions] = update_state_and_transition(states_r, states, transitions, direction_no, learning_rate, discount_factor, agent_loc, agent_loc_past, target_value, cat_value, softmax_func);

                    if agent_loc == target_loc
                        reach_target_bool = 1;
                        states(target_loc(1,1), target_loc(1,2)) = target_value;
                    end
                    if agent_loc == cat_loc
                        reach_cat_bool = 1;
                        states(cat_loc(1,1), cat_loc(1,2)) = cat_value;
                    end

                    step_no = step_no + 1;
                    agent_loc_past = agent_loc;

                    if step_no>max_steps
                        break
                    end
                end
                num_steps(iter_no, trial_no) = step_no;
            end
        end
        num_steps = mean(num_steps, 1);
        steps_mat(lr_index, df_index) = mean(num_steps(end-50:end));
    end
end

imagesc(learning_rates, discount_factors, steps_mat)
set(gca,'YDir','normal')
ylabel('Learning Rate')
xlabel('Discount Factor')
c = colorbar;
colormap bone
c.Label.String = 'Avg number of steps to reach the target';
if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/Q03",'-dpng','-r0')
end
%% Effect of learning factor and discount factor - 2 targets
clc
clear
close all

save_figures = 1;

learning_rates = 0.1:0.3:1;
discount_factors = 0.1:0.3:1;

% configs
target_value = 2;
cat_value = -2;
num_trials = 400;
num_iters = 10;
forgetting_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [12, 13];
target_loc_1 = [7, 7];
target_loc_2 = [4, 10];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc_1(1,1), target_loc_1(1,2)) = 1;
map_mat_main(target_loc_2(1,1), target_loc_2(1,2)) = 1;

num_directions = 4;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

steps_mat = zeros(length(learning_rates), length(discount_factors));

num_steps = zeros(num_iters, num_trials);
lr_index = 0;

for learning_rate = learning_rates
    learning_rate
    lr_index = lr_index+1;
    df_index = 0;
    for discount_factor = discount_factors
        discount_factor
        df_index = df_index+1;
        for iter_no = 1:num_iters
            % initializing the movement direction probabilities
            states = zeros(map_size);

            for i = 1:map_size(1,1)
                for j = 1:map_size(1,2)
                    transitions{i,j} = zeros(1,4);
                end
            end

            % defining the softmax function
            softmax_func = @(x) exp(x)/sum(exp(x));


            states_r = zeros(map_size);
            states_r(target_loc_1(1,1), target_loc_1(1,2)) = target_value;
            states_r(target_loc_2(1,1), target_loc_2(1,2)) = target_value;
            states_r(cat_loc(1,1), cat_loc(1,2)) = cat_value;

            for trial_no = 1:num_trials
                % applying the forgeting factor
                states = states*forgetting_factor;

                reach_target_bool = 0;
                reach_cat_bool = 0;
                agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                agent_loc_past = [1, 1];
                step_no = 1;
                agent_locs = [];    

                while ~reach_target_bool && ~reach_cat_bool


                    % deciding which direction to choose
                    directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))));
                    direction_no = choose_by_prob(directions_probs);

                    direction = direction_map{direction_no};
                    check_mat = map_size - (agent_loc + direction);
                    if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                       agent_loc = agent_loc + direction;
                    else
                        for i = randperm(num_directions)
                            direction = direction_map{i};
                            check_mat = map_size - (agent_loc + direction);
                            if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                                agent_loc = agent_loc + direction;
                                pass_bool = 1;
                                break
                            end
                        end
                        if ~pass_bool
                            agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                        end
                    end

                    % checking location

                    agent_locs = [agent_locs; agent_loc];

                    [states, transitions] = update_state_and_transition(states_r, states, transitions, direction_no, learning_rate, discount_factor, agent_loc, agent_loc_past, target_value, cat_value, softmax_func);

                    if agent_loc == target_loc_1 | agent_loc == target_loc_2
                        reach_target_bool = 1;
                        states(target_loc_1(1,1), target_loc_1(1,2)) = target_value;
                        states(target_loc_2(1,1), target_loc_2(1,2)) = target_value;
                    end
                    if agent_loc == cat_loc
                        reach_cat_bool = 1;
                        states(cat_loc(1,1), cat_loc(1,2)) = cat_value;
                    end

                    step_no = step_no + 1;
                    agent_loc_past = agent_loc;

                    if step_no>max_steps
                        break
                    end
                end
                num_steps(iter_no, trial_no) = step_no;
            end
        end
        num_steps = mean(num_steps, 1);
        steps_mat(lr_index, df_index) = mean(num_steps(end-10:end));
    end
end

imagesc(learning_rates, discount_factors, steps_mat)
set(gca,'YDir','normal')
ylabel('Learning Rate')
xlabel('Discount Factor')
c = colorbar;
colormap bone
c.Label.String = 'Avg number of steps to reach the target';
if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/Q04",'-dpng','-r0')
end
%% TD Rule
clc
clear
close all

save_figures = 0;
show_plot = 0;
save_freq = 1;
video_boolian = 1;

% loading images
rat_img = imread('assets/rat.png');
cat_img = imread('assets/cat.png');
target_img = imread('assets/target.png');

% configs
target_value = 2;
cat_value = -2;
num_trials = 1000;
learning_rate_main = 0.1;
forgetting_factor = 1;
discount_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
cat_loc = [12, 13];
target_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
target_loc = [7, 7];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_directions = 4;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

% initializing the movement direction probabilities
states = zeros(map_size);

for i = 1:map_size(1,1)
    for j = 1:map_size(1,2)
        transitions{i,j} = zeros(1,4);
    end
end

% defining the softmax function
softmax_func = @(x) exp(x)/sum(exp(x));

fig = figure('units','normalized','outerposition',[0 0 1 1]);


if video_boolian
    writerObj = VideoWriter("videos/Demo_TD_Rule_4Direction");
    writerObj.FrameRate = 50;
    writerObj.Quality = 20;
    open(writerObj);
end

states_r = zeros(map_size);
states_r(target_loc(1,1), target_loc(1,2)) = target_value;
states_r(cat_loc(1,1), cat_loc(1,2)) = cat_value;
num_steps = zeros(1, num_trials);
for trial_no = 1:num_trials
    % applying the forgeting factor
    states = states*forgetting_factor;

    learning_rate = learning_rate_main/(1+log10(trial_no));
    reach_target_bool = 0;
    reach_cat_bool = 0;
    agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
    agent_loc_past = [1, 1];
    step_no = 1;
    agent_locs = [agent_loc_past];
    direction_nos = [1];
    
    if mod(trial_no-1, save_freq) == 0
        show_plot = 1;
    else
        show_plot = 0;
    end
    
    while ~reach_target_bool && ~reach_cat_bool
        

        % deciding which direction to choose
        directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))))
        direction_no = choose_by_prob(directions_probs)
        
        direction = direction_map{direction_no};
        check_mat = map_size - (agent_loc + direction);
        if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
           agent_loc = agent_loc + direction;
        else
            for i = randperm(num_directions)
                direction = direction_map{i};
                check_mat = map_size - (agent_loc + direction);
                if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                    agent_loc = agent_loc + direction;
                    pass_bool = 1;
                    break
                end
            end
            if ~pass_bool
                agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
            end
        end

        % checking location
        
        agent_locs = [agent_locs; agent_loc];
        direction_nos = [direction_nos; direction_no];
        [states, transitions] = update_state_and_transition_TD(states_r, states, transitions, direction_nos, learning_rate, discount_factor, target_value, cat_value, softmax_func, agent_locs);
        
        if agent_loc == target_loc
            reach_target_bool = 1;
            states(target_loc(1,1), target_loc(1,2)) = target_value;
        end
        if agent_loc == cat_loc
            reach_cat_bool = 1;
            states(cat_loc(1,1), cat_loc(1,2)) = cat_value;
        end

        if show_plot
            subplot(2,3,[1,2,4,5])
            plot_map(agent_loc, agent_locs(2:end, :), target_loc, cat_loc, map_size, rat_img, cat_img, target_img)
            title("Trial "+num2str(trial_no)+" | "+"Step "+num2str(step_no)+" | LR "+num2str(learning_rate))
            subplot(2,3,3)
            imagesc(states);
            set(gca,'YDir','normal')
            colormap bone
            colorbar
            caxis([cat_value, target_value])
            title('States')
            subplot(2,3,6)
            [fx,fy] = gradient(states);
            quiver(fx,fy)
            title('States Gradients')            
        end
        
        step_no = step_no + 1;
        agent_loc_past = agent_loc;
        
        if video_boolian
            frame = getframe(fig);
            for frame_index = 1:2
                writeVideo(writerObj,frame);
            end
        else
            pause(0.01)
        end
        
        if step_no>max_steps
            break
        end
    end
    num_steps(trial_no) = step_no;
    if mod(trial_no-1, save_freq) == 0
        if save_figures
            set(gcf,'PaperPositionMode','auto')
            print("images/trial_TD_"+num2str(trial_no),'-dpng','-r0')
        end
    end
end

if video_boolian
    close(writerObj)
end
%% Step numbers vs trials - TD rule
clc
clear
close all

save_figures = 1;

% configs
target_value = 2;
cat_value = -2;
num_trials = 1001;
trial_step = 2;
num_iters = 100;
learning_rate_main = 0.1;
forgetting_factor = 1;
discount_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
cat_loc = [12, 13];
target_loc = [ceil(map_size(1,1)*rand(1,1)), ceil(map_size(1,2)*rand(1,1))];
target_loc = [7, 7];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_directions = 4;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

num_steps = zeros(num_iters, num_trials);

for iter_no = 1:num_iters
    iter_no
    % initializing the movement direction probabilities
    states = zeros(map_size);

    for i = 1:map_size(1,1)
        for j = 1:map_size(1,2)
            transitions{i,j} = zeros(1,4);
        end
    end

    % defining the softmax function
    softmax_func = @(x) exp(x)/sum(exp(x));


    states_r = zeros(map_size);
    states_r(target_loc(1,1), target_loc(1,2)) = target_value;
    states_r(cat_loc(1,1), cat_loc(1,2)) = cat_value;

    for trial_no = 1:trial_step:num_trials
        % applying the forgeting factor
        states = states*forgetting_factor;

        learning_rate = learning_rate_main/(1+log10(trial_no));
        reach_target_bool = 0;
        reach_cat_bool = 0;
        agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
        agent_loc_past = [1, 1];
        step_no = 1;
        agent_locs = [agent_loc_past];
        direction_nos = [1];

        while ~reach_target_bool && ~reach_cat_bool


            % deciding which direction to choose
            directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))));
            direction_no = choose_by_prob(directions_probs);

            direction = direction_map{direction_no};
            check_mat = map_size - (agent_loc + direction);
            if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
               agent_loc = agent_loc + direction;
            else
                for i = randperm(num_directions)
                    direction = direction_map{i};
                    check_mat = map_size - (agent_loc + direction);
                    if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                        agent_loc = agent_loc + direction;
                        pass_bool = 1;
                        break
                    end
                end
                if ~pass_bool
                    agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                end
            end

            % checking location

            agent_locs = [agent_locs; agent_loc];

            direction_nos = [direction_nos; direction_no];
            [states, transitions] = update_state_and_transition_TD(states_r, states, transitions, direction_nos, learning_rate, discount_factor, target_value, cat_value, softmax_func, agent_locs);
            
            if agent_loc == target_loc
                reach_target_bool = 1;
                states(target_loc(1,1), target_loc(1,2)) = target_value;
            end
            if agent_loc == cat_loc
                reach_cat_bool = 1;
                states(cat_loc(1,1), cat_loc(1,2)) = cat_value;
            end

            step_no = step_no + 1;
            agent_loc_past = agent_loc;

            if step_no>max_steps
                break
            end
        end
        num_steps(iter_no, trial_no) = step_no;
    end
end

plot(1:trial_step:num_trials, mean(num_steps(:, 1:trial_step:num_trials), 1), 'k', 'LineWidth', 1)
xlabel('Trial No')
ylabel('Num of Steps')
title('Num of steps to reach target vs Trials')
xlim([1, num_trials])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/Q05",'-dpng','-r0')
end

%% TD vs simple RL
clc
close all
save_figures = 1;
figure
plot(1:trial_step:num_trials, mean(num_steps(:, 1:trial_step:num_trials), 1), 'k', 'LineWidth', 1)
hold on
plot(1:trial_step:num_trials, mean(num_steps_rl(:, 1:trial_step:num_trials), 1), 'b', 'LineWidth', 1)
xlabel('Trial No')
ylabel('Num of Steps')
legend('TD rule', 'Simple RL')
title('Num of steps to reach target vs Trials')
xlim([1, num_trials])

if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/Q05_2",'-dpng','-r0')
end


%% Effect of learning factor and discount factor - TD Rule
clc
clear
close all

save_figures = 1;

learning_rates = 0.1:0.3:1;
discount_factors = 0.1:0.3:1;

% configs
target_value = 2;
cat_value = -2;
num_trials = 400;
num_iters = 10;
forgetting_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [12, 13];
target_loc = [7, 7];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_directions = 4;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right

steps_mat = zeros(length(learning_rates), length(discount_factors));

num_steps = zeros(num_iters, num_trials);
lr_index = 0;

for learning_rate = learning_rates
    learning_rate
    lr_index = lr_index+1;
    df_index = 0;
    for discount_factor = discount_factors
        discount_factor
        df_index = df_index+1;
        for iter_no = 1:num_iters
            % initializing the movement direction probabilities
            states = zeros(map_size);

            for i = 1:map_size(1,1)
                for j = 1:map_size(1,2)
                    transitions{i,j} = zeros(1,4);
                end
            end

            % defining the softmax function
            softmax_func = @(x) exp(x)/sum(exp(x));


            states_r = zeros(map_size);
            states_r(target_loc(1,1), target_loc(1,2)) = target_value;
            states_r(cat_loc(1,1), cat_loc(1,2)) = cat_value;

            for trial_no = 1:num_trials
                % applying the forgeting factor
                states = states*forgetting_factor;

                reach_target_bool = 0;
                reach_cat_bool = 0;
                agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                agent_loc_past = [1, 1];
                step_no = 1;
                agent_locs = [agent_loc_past];
                direction_nos = [1]; 

                while ~reach_target_bool && ~reach_cat_bool


                    % deciding which direction to choose
                    directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))));
                    direction_no = choose_by_prob(directions_probs);

                    direction = direction_map{direction_no};
                    check_mat = map_size - (agent_loc + direction);
                    if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                       agent_loc = agent_loc + direction;
                    else
                        for i = randperm(num_directions)
                            direction = direction_map{i};
                            check_mat = map_size - (agent_loc + direction);
                            if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                                agent_loc = agent_loc + direction;
                                pass_bool = 1;
                                break
                            end
                        end
                        if ~pass_bool
                            agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                        end
                    end

                    % checking location

                    agent_locs = [agent_locs; agent_loc];

                    direction_nos = [direction_nos; direction_no];
                    [states, transitions] = update_state_and_transition_TD(states_r, states, transitions, direction_nos, learning_rate, discount_factor, target_value, cat_value, softmax_func, agent_locs);
            

                    if agent_loc == target_loc
                        reach_target_bool = 1;
                        states(target_loc(1,1), target_loc(1,2)) = target_value;
                    end
                    if agent_loc == cat_loc
                        reach_cat_bool = 1;
                        states(cat_loc(1,1), cat_loc(1,2)) = cat_value;
                    end

                    step_no = step_no + 1;
                    agent_loc_past = agent_loc;

                    if step_no>max_steps
                        break
                    end
                end
                num_steps(iter_no, trial_no) = step_no;
            end
        end
        num_steps = mean(num_steps, 1);
        steps_mat(lr_index, df_index) = mean(num_steps(end-50:end));
    end
end

imagesc(learning_rates, discount_factors, steps_mat)
set(gca,'YDir','normal')
ylabel('Learning Rate')
xlabel('Discount Factor')
c = colorbar;
colormap bone
c.Label.String = 'Avg number of steps to reach the target';
if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/Q05_3_2",'-dpng','-r0')
end

%% TD Rule - 8 directions
clc
clear
close all

save_figures = 0;
show_plot = 0;
save_freq = 1;
video_boolian = 1;

% loading images
rat_img = imread('assets/rat.png');
cat_img = imread('assets/cat.png');
target_img = imread('assets/target.png');

% configs
target_value = 2;
cat_value = -2;
num_trials = 500;
learning_rate_main = 0.1;
forgetting_factor = 1;
discount_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [12, 13];
target_loc = [7, 7];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_directions = 8;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right
direction_map{5} = [1, 1]; % up right
direction_map{6} = [-1, 1]; % up left
direction_map{7} = [-1, -1]; % down left 
direction_map{8} = [1, -1]; % down right

% initializing the movement direction probabilities
states = zeros(map_size);

for i = 1:map_size(1,1)
    for j = 1:map_size(1,2)
        transitions{i,j} = zeros(1,num_directions);
    end
end

% defining the softmax function
softmax_func = @(x) exp(x)/sum(exp(x));

fig = figure('units','normalized','outerposition',[0 0 1 1]);

if video_boolian
    writerObj = VideoWriter("videos/Demo_TD_Rule_8Direction");
    writerObj.FrameRate = 50;
    writerObj.Quality = 20;
    open(writerObj);
end


states_r = zeros(map_size);
states_r(target_loc(1,1), target_loc(1,2)) = target_value;
states_r(cat_loc(1,1), cat_loc(1,2)) = cat_value;
num_steps = zeros(1, num_trials);
for trial_no = 1:num_trials
    % applying the forgeting factor
    states = states*forgetting_factor;

    learning_rate = learning_rate_main/(1+log10(trial_no));
    reach_target_bool = 0;
    reach_cat_bool = 0;
    agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
    agent_loc_past = [1, 1];
    step_no = 1;
    agent_locs = [agent_loc_past];
    direction_nos = [1];
    
    if mod(trial_no-1, save_freq) == 0
        show_plot = 1;
    else
        show_plot = 0;
    end
    
    while ~reach_target_bool && ~reach_cat_bool
        

        % deciding which direction to choose
        directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))))
        direction_no = choose_by_prob(directions_probs)
        
        direction = direction_map{direction_no};
        check_mat = map_size - (agent_loc + direction);
        if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
           agent_loc = agent_loc + direction;
        else
            for i = randperm(num_directions)
                direction = direction_map{i};
                check_mat = map_size - (agent_loc + direction);
                if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                    agent_loc = agent_loc + direction;
                    pass_bool = 1;
                    break
                end
            end
            if ~pass_bool
                agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
            end
        end

        % checking location
        
        agent_locs = [agent_locs; agent_loc];
        direction_nos = [direction_nos; direction_no];
        [states, transitions] = update_state_and_transition_TD(states_r, states, transitions, direction_nos, learning_rate, discount_factor, target_value, cat_value, softmax_func, agent_locs);
        
        if agent_loc == target_loc
            reach_target_bool = 1;
            states(target_loc(1,1), target_loc(1,2)) = target_value;
        end
        if agent_loc == cat_loc
            reach_cat_bool = 1;
            states(cat_loc(1,1), cat_loc(1,2)) = cat_value;
        end

        if show_plot
            subplot(2,3,[1,2,4,5])
            plot_map(agent_loc, agent_locs(2:end, :), target_loc, cat_loc, map_size, rat_img, cat_img, target_img)
            title("Trial "+num2str(trial_no)+" | "+"Step "+num2str(step_no)+" | LR "+num2str(learning_rate))
            subplot(2,3,3)
            imagesc(states);
            set(gca,'YDir','normal')
            colormap bone
            colorbar
            caxis([cat_value, target_value])
            title('States')
            subplot(2,3,6)
            [fx,fy] = gradient(states);
            quiver(fx,fy)
            title('States Gradients')            
        end
        
        step_no = step_no + 1;
        agent_loc_past = agent_loc;
        
        if video_boolian
            frame = getframe(fig);
            for frame_index = 1:2
                writeVideo(writerObj,frame);
            end
        else
            pause(0.01)
        end
        
        if step_no>max_steps
            break
        end
    end
    num_steps(trial_no) = step_no;
    if mod(trial_no-1, save_freq) == 0
        if save_figures
            set(gcf,'PaperPositionMode','auto')
            print("images/trial_TD_8d_"+num2str(trial_no),'-dpng','-r0')
        end
    end
end

if video_boolian
    close(writerObj)
end
%% Effect of learning factor and discount factor - TD Rule with 8 directions
clc
clear
close all

save_figures = 1;

learning_rates = 0.1:0.3:1;
discount_factors = 0.1:0.3:1;

% configs
target_value = 2;
cat_value = -2;
num_trials = 200;
num_iters = 5;
forgetting_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [12, 13];
target_loc = [7, 7];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_directions = 8;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right
direction_map{5} = [1, 1]; % up right
direction_map{6} = [-1, 1]; % up left
direction_map{7} = [-1, -1]; % down left 
direction_map{8} = [1, -1]; % down right

steps_mat = zeros(length(learning_rates), length(discount_factors));

num_steps = zeros(num_iters, num_trials);
lr_index = 0;

for learning_rate = learning_rates
    learning_rate
    lr_index = lr_index+1;
    df_index = 0;
    for discount_factor = discount_factors
        discount_factor
        df_index = df_index+1;
        for iter_no = 1:num_iters
            % initializing the movement direction probabilities
            states = zeros(map_size);

            for i = 1:map_size(1,1)
                for j = 1:map_size(1,2)
                    transitions{i,j} = zeros(1,num_directions);
                end
            end

            % defining the softmax function
            softmax_func = @(x) exp(x)/sum(exp(x));


            states_r = zeros(map_size);
            states_r(target_loc(1,1), target_loc(1,2)) = target_value;
            states_r(cat_loc(1,1), cat_loc(1,2)) = cat_value;

            for trial_no = 1:num_trials
                % applying the forgeting factor
                states = states*forgetting_factor;

                reach_target_bool = 0;
                reach_cat_bool = 0;
                agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                agent_loc_past = [1, 1];
                step_no = 1;
                agent_locs = [agent_loc_past];
                direction_nos = [1]; 

                while ~reach_target_bool && ~reach_cat_bool


                    % deciding which direction to choose
                    directions_probs = softmax_func(cell2mat(transitions(agent_loc(1,1), agent_loc(1,2))));
                    
                    direction_no = choose_by_prob(directions_probs);

                    direction = direction_map{direction_no};
                    check_mat = map_size - (agent_loc + direction);
                    if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                       agent_loc = agent_loc + direction;
                    else
                        for i = randperm(num_directions)
                            direction = direction_map{i};
                            check_mat = map_size - (agent_loc + direction);
                            if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                                agent_loc = agent_loc + direction;
                                pass_bool = 1;
                                break
                            end
                        end
                        if ~pass_bool
                            agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
                        end
                    end

                    % checking location

                    agent_locs = [agent_locs; agent_loc];

                    direction_nos = [direction_nos; direction_no];
                    [states, transitions] = update_state_and_transition_TD(states_r, states, transitions, direction_nos, learning_rate, discount_factor, target_value, cat_value, softmax_func, agent_locs);
            

                    if agent_loc == target_loc
                        reach_target_bool = 1;
                        states(target_loc(1,1), target_loc(1,2)) = target_value;
                    end
                    if agent_loc == cat_loc
                        reach_cat_bool = 1;
                        states(cat_loc(1,1), cat_loc(1,2)) = cat_value;
                    end

                    step_no = step_no + 1;
                    agent_loc_past = agent_loc;

                    if step_no>max_steps
                        break
                    end
                end
                num_steps(iter_no, trial_no) = step_no;
            end
        end
        num_steps = mean(num_steps, 1);
        steps_mat(lr_index, df_index) = mean(num_steps(end-50:end));
    end
end

imagesc(learning_rates, discount_factors, steps_mat)
set(gca,'YDir','normal')
ylabel('Learning Rate')
xlabel('Discount Factor')
c = colorbar;
colormap bone
c.Label.String = 'Avg number of steps to reach the target';
if save_figures
    set(gcf,'PaperPositionMode','auto')
    print("Report/images/Q05_4",'-dpng','-r0')
end

%% cat and agent both learn state's values
clc
clear
close all

save_figures = 0;
show_plot = 0;
save_freq = 50;
video_boolian = 0;

% loading images
rat_img = imread('assets/rat.png');
cat_img = imread('assets/cat.png');
target_img = imread('assets/target.png');

% configs
target_value = 2;
cat_value = -2;
num_trials = 500;
learning_rate_main = 0.1;
forgetting_factor = 1;
discount_factor = 1;
max_steps = 200;

map_size = [15, 15];
cat_loc = [12, 13];
target_loc = [7, 7];

map_mat_main = zeros(map_size);
map_mat_main(cat_loc(1,1), cat_loc(1,2)) = -1;
map_mat_main(target_loc(1,1), target_loc(1,2)) = 1;

num_directions = 8;

direction_map{1} = [0, 1]; % up
direction_map{2} = [0, -1]; % down
direction_map{3} = [-1, 0]; % left
direction_map{4} = [1, 0]; % right
direction_map{5} = [1, 1]; % up right
direction_map{6} = [-1, 1]; % up left
direction_map{7} = [-1, -1]; % down left 
direction_map{8} = [1, -1]; % down right

% initializing the movement direction probabilities
states_agent = zeros(map_size);
states_cat = zeros(map_size);

for i = 1:map_size(1,1)
    for j = 1:map_size(1,2)
        transitions_agent{i,j} = zeros(1,num_directions);
        transitions_cat{i,j} = zeros(1,num_directions);
    end
end

% defining the softmax function
softmax_func = @(x) exp(x)/sum(exp(x));

fig = figure('units','normalized','outerposition',[0 0 1 1]);

if video_boolian
    writerObj = VideoWriter("videos/Demo_TD_Rule_8Direction_InteligentCat");
    writerObj.FrameRate = 50;
    writerObj.Quality = 20;
    open(writerObj);
end


states_r = zeros(map_size);
states_r(target_loc(1,1), target_loc(1,2)) = target_value;

num_steps = zeros(1, num_trials);

for trial_no = 1:num_trials
    % applying the forgeting factor
    states_agent = states_agent*forgetting_factor;
    states_cat = states_cat*forgetting_factor;

    learning_rate = learning_rate_main/(1+log10(trial_no));
    agent_reach_target_bool = 0;
    agent_reach_cat_bool = 0;
    cat_reach_agent_bool = 0;
    agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
    cat_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
    agent_loc_past = [1, 1];
    cat_loc_past = [15,15];
    step_no = 1;
    agent_locs = [agent_loc_past];
    cat_locs = [cat_loc_past];
    direction_nos_agent = [1];
    direction_nos_cat = [1];
    
    if mod(trial_no-1, save_freq) == 0
        show_plot = 1;
    else
        show_plot = 0;
    end
    
    while ~agent_reach_target_bool && ~agent_reach_cat_bool && ~cat_reach_agent_bool
        states_r_cat = zeros(map_size);
        states_r_cat(agent_loc(1,1), agent_loc(1,2)) = target_value;
        
        % deciding which direction to choose
        directions_probs_agent = softmax_func(cell2mat(transitions_agent(agent_loc(1,1), agent_loc(1,2))));
        direction_no_agent = choose_by_prob(directions_probs_agent);
        
        directions_probs_cat = softmax_func(cell2mat(transitions_cat(agent_loc(1,1), agent_loc(1,2))));
        direction_no_cat = choose_by_prob(directions_probs_cat);
        
        direction_agent = direction_map{direction_no_agent};
        check_mat = map_size - (agent_loc + direction_agent);
        if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
           agent_loc = agent_loc + direction_agent;
        else
            for i = randperm(num_directions)
                direction_agent = direction_map{i};
                check_mat = map_size - (agent_loc + direction_agent);
                if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                    agent_loc = agent_loc + direction_agent;
                    pass_bool = 1;
                    break
                end
            end
            if ~pass_bool
                agent_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
            end
        end
        
        direction_cat = direction_map{direction_no_cat};
        check_mat = map_size - (cat_loc + direction_cat);
        if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
           cat_loc = cat_loc + direction_cat;
        else
            for i = randperm(num_directions)
                direction_cat = direction_map{i};
                check_mat = map_size - (cat_loc + direction_cat);
                if check_mat(1,1) > 0 && check_mat(1,2) > 0 && check_mat(1,1) < map_size(1,1) && check_mat(1,2) < map_size(1,2)
                    cat_loc = cat_loc + direction_cat;
                    pass_bool = 1;
                    break
                end
            end
            if ~pass_bool
                cat_loc = [randi(map_size(1,1), 1, 1), randi(map_size(1,2), 1, 1)];
            end
        end

        % checking location
        
        agent_locs = [agent_locs; agent_loc];
        direction_nos_agent = [direction_nos_agent; direction_no_agent];
        [states_agent, transitions_agent] = update_state_and_transition_TD(states_r, states_agent, ...
                                            transitions_agent, direction_nos_agent, learning_rate, ...
                                            discount_factor, target_value, cat_value, softmax_func, agent_locs);
                                        
                                        
        cat_locs = [cat_locs; cat_loc];
        direction_nos_cat = [direction_nos_cat; direction_no_cat];
        [states_cat, transitions_cat] = update_state_and_transition_TD(states_r_cat, states_cat, ...
                                            transitions_cat, direction_nos_cat, learning_rate, ...
                                            discount_factor, target_value, cat_value, softmax_func, cat_locs);
        
        if agent_loc == target_loc
            agent_reach_target_bool = 1;
            states_agent(target_loc(1,1), target_loc(1,2)) = target_value;
        end
        
        if agent_loc == cat_loc
                                        
            agent_reach_cat_bool = 1;
            cat_reach_agent_loc = 1;
            states_agent(cat_loc(1,1), cat_loc(1,2)) = cat_value;
            states_cat(agent_loc(1,1), agent_loc(1,2)) = target_value;
        end

        if show_plot
            subplot(2,3,[1,2,4,5])
            plot_map(agent_loc, agent_locs(2:end, :), target_loc, cat_loc, map_size, rat_img, cat_img, target_img)
            title("Trial "+num2str(trial_no)+" | "+"Step "+num2str(step_no)+" | LR "+num2str(learning_rate))
            subplot(2,3,3)
            imagesc(states_agent);
            set(gca,'YDir','normal')
            colormap bone
            colorbar
            caxis([cat_value, target_value])
            title('States Agent')
            subplot(2,3,6)
            imagesc(states_cat);
            set(gca,'YDir','normal')
            colormap bone
            colorbar
            caxis([cat_value, target_value])
            title('States Cat')         
        end
        
        step_no = step_no + 1;
        agent_loc_past = agent_loc;
        
        if video_boolian
            frame = getframe(fig);
            for frame_index = 1:2
                writeVideo(writerObj,frame);
            end
        else
            pause(0.01)
        end
        
        if step_no>max_steps
            break
        end
    end
    num_steps(trial_no) = step_no;
    if mod(trial_no-1, save_freq) == 0
        if save_figures
            set(gcf,'PaperPositionMode','auto')
            print("images/trial_TD_8d_"+num2str(trial_no),'-dpng','-r0')
        end
    end
end

if video_boolian
    close(writerObj)
end
