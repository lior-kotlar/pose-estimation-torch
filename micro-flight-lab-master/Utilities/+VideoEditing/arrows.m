function arrows()

%     px_to_mm=0.057;
    px_to_mm=1;
%     full_name='D:\Movies\2017_10_04_Asian_tiger\dimensions_1px=57um.jpg';
    [fname,path,FilterIndex] = uigetfile('*.*','Select Pic');
    full_n;lame=[path,fname];
    barename=strsplit(fname,'.');
    new_fname=[barename{1} '_arrow.jpg'];
    rawData1 = importdata(full_name);
    fig=figure;
    ax=axes(fig);
    
    imagesc(rawData1);
    hold on
    
    arrow_btn = uicontrol('Style', 'pushbutton', 'String', 'Arrow',...
            'Position', [20 20 50 20],...
            'Callback', @putarrow);
    save_btn = uicontrol('Style', 'pushbutton', 'String', 'Save',...
            'Position', [20 50 50 20],...
            'Callback', @save_exit);
    
    zoom on
    
    function putarrow(hObject,eventdata,handles)
        zoom off
        pts = ginput(2);
        p1 = pts(1,:);
        p2 = pts(2,:);
        dp = p2-p1;
        hold on
        a_middle=mean(pts,1);
        text(a_middle(1),a_middle(2), [num2str(norm(dp)*px_to_mm),'mm'],'Color','red','BackgroundColor','white',...
            'Margin',0.05);
        quiver(p1(1),p1(2),dp(1),dp(2),0,'Color','red','MaxHeadSize',0.4)
        quiver(p2(1),p2(2),-dp(1),-dp(2),0,'Color','red','MaxHeadSize',0.4)
    end

    function save_exit(hObject,eventdata,handles)
        frame=getframe(ax);
        realframe=frame.cdata;
        imwrite(realframe,[path,new_fname])
        close(fig);
    end
end