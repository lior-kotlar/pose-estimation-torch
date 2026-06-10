function figureplayer()
% Description:
% plays figures loaded from a folder given by user

    global handles_struct fig_filenames
    
    folder_name =uigetdir();
    if isequal(folder_name,0)
       disp('User selected Cancel')
       return
    end
    
    listing = dir(folder_name);
    Filenames={listing.name};
    fig_filenames=Filenames(cell2mat(cellfun(@(x) contains(x,'.fig'),...
        Filenames,'UniformOutput',0)));
    Nframes=numel(fig_filenames);
    
    handles_struct.hFig = openfig(fullfile(folder_name,fig_filenames{1}));
    handles_struct.hax = findobj(handles_struct.hFig,'Type','axes');
    handles_struct.sld = uicontrol(handles_struct.hFig,'Style', 'slider',...
        'Units','Normalized',...
        'Min',1,'Max',Nframes,'Value',1,...
        'SliderStep',[1,10]/(double(Nframes)-1),...
        'Position', [0.01 0.01 0.9 0.03],'Enable','on');
    addlistener(handles_struct.sld, 'Value', 'PostSet',@cont_sld);
    
    function cont_sld(~, ~)
        frame_number = round(handles_struct.sld.Value);
        disp(frame_number)
        handles_struct.sld.Value=frame_number;
        cla(handles_struct.hax)
        load_f = openfig(fullfile(folder_name,fig_filenames{frame_number}),'invisible');
        load_a=findobj(load_f,'Type','axes');
        copyobj(get(load_a,'Children'),handles_struct.hax)
        title(load_a.Title.String,'Parent',handles_struct.hax)
        delete(load_f)
    end
end