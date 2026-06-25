function PlotPath(startPose,path,rmin)
    type = path.type;
    x = [];
    y = [];
    angle=[];
    seg = [path.t,path.u,path.v,path.w,path.x];
    pvec = startPose;
    for i = 1:5        
        if type(i) == 'S'
            theta = pvec(3);
            t = pi/2+linspace(theta,theta);
            angle=[angle,t];
            dl = rmin*seg(i);
            dvec = [dl*cos(theta), dl*sin(theta), 0];
            dx = pvec(1)+linspace(0,dvec(1));
            dy = pvec(2)+linspace(0,dvec(2));
            x = [x,dx];
            y = [y,dy];
            pvec = pvec+dvec;
        elseif type(i) == 'L'
            theta = pvec(3);
            dtheta = seg(i);
            cenx = pvec(1)-rmin*sin(theta);
            ceny = pvec(2)+rmin*cos(theta);
            t = theta-pi/2+linspace(0,dtheta);
            dx = cenx+rmin*cos(t);
            dy = ceny+rmin*sin(t);
            x = [x,dx];
            y = [y,dy];
            angle=[angle,t];
            theta = theta+dtheta;
            pvec = [dx(end),dy(end),theta];
            dl = dtheta;
        elseif type(i) == 'R'
            theta = pvec(3);
            dtheta = -seg(i);
            cenx = pvec(1)+rmin*sin(theta);
            ceny = pvec(2)-rmin*cos(theta);
            t = theta+pi/2+linspace(0,dtheta);
            dx = cenx+rmin*cos(t);
            dy = ceny+rmin*sin(t);
            x = [x,dx];
            y = [y,dy];
            angle=[angle,t];
            theta = theta+dtheta;
            pvec = [dx(end),dy(end),theta];
            dl = -dtheta;
        else
            % do nothing
        end
        if dl > 0
            plot(dx,dy,'b');
        else
            plot(dx,dy,'r');
        end
        hold on
    end
    axis equal
    plot(startPose(1),startPose(2),'kx','LineWidth',2,'MarkerSize',10)
    plot(x(end),y(end),'ko', 'LineWidth',2,'MarkerSize',10)
%     veh = plot(x(1),y(1),'d','MarkerFaceColor','g','MarkerSize',10);
    videoFWriter = VideoWriter('Parking1.mp4','MPEG-4');
    open(videoFWriter);
    [vehx,vehy] = getVehTran(x(1),y(1),angle(1)); % 根据后轴中心的位姿计算车辆边框的位姿
    h1 = plot(vehx,vehy,'r','LineWidth',4); % 车辆边框
    h2 = plot(x(1),y(1),'rx','MarkerSize',10); % 车辆后轴中心
    img = getframe(gcf);
    hold off
    pause(1)
    for k = 2:length(x)
        veh.XData = x(k);
        veh.YData = y(k);
        angle_2=angle(k);
        dl = norm([x(k)-x(k-1),y(k)-y(k-1)]);
        [vehx,vehy] = getVehTran(veh.XData,veh.YData,angle_2);
        h1.XData = vehx; % 更新h1图像句柄,把车辆边框四个角点的x坐标添加进去
        h1.YData = vehy;
        h2.XData = veh.XData; % 更新h2图像句柄,把车辆边框四个角点的y坐标添加进去
        h2.YData = veh.YData;
        writeVideo(videoFWriter,img);
        pause(dl)
    end    
end