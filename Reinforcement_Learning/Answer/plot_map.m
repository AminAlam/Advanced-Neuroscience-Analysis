function plot_map(agent_loc, target_loc, cat_loc, map_size, rat_img, cat_img, target_img)
    scatter(agent_loc(1,2), agent_loc(1,1), 'k', 'filled')
    hold on
    image([agent_loc(1,2)-0.5, agent_loc(1,2)+0.5], [agent_loc(1,1)-0.5, agent_loc(1,1)+0.5], rat_img)
    
    scatter(target_loc(1,2), target_loc(1,1), 'b', 'filled')
    image([target_loc(1,2)-0.5, target_loc(1,2)+0.5], [target_loc(1,1)-0.5, target_loc(1,1)+0.5], target_img)
    
    scatter(cat_loc(1,2), cat_loc(1,1), 'r', 'filled')
    image([cat_loc(1,2)-0.5, cat_loc(1,2)+0.5], [cat_loc(1,1)-0.5, cat_loc(1,1)+0.5], cat_img)
    
    hold off
    xlim([0, map_size(1,1)+1])
    ylim([0, map_size(1,2)+1])
end