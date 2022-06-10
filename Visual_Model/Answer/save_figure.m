function save_figure(file_name)
    set(gcf,'PaperPositionMode','auto')
    print(file_name,'-dpng','-r0')
end