function direction_no = choose_by_prob(directions_probs)
    rand_num = rand()
    wood = [0];
    counter = 1;
    for i = directions_probs
        wood = [wood, wood(end)+i]
        if rand_num > wood(end-1) && rand_num < wood(end)
            direction_no = counter;
            return
        end
        counter = counter+1;
    end 

end