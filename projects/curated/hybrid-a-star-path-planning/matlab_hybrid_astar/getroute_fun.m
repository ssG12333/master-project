function route=getroute_fun(startPose,path,rmin)
type = path.type;
x = [];
y = [];
angle=[];
n=50;
seg = [path.t,path.u,path.v,path.w,path.x];
pvec = startPose;
for i = 1:5
    if type(i) == 'S'
        theta = pvec(3);
        t = linspace(theta,theta,fix(abs(seg(i))*n)+2);
        angle=[angle,t];
        dl = rmin*seg(i);
        if seg(i)==0
            continue
        end
        dvec = [dl*cos(theta), dl*sin(theta), 0];
        dx = pvec(1)+linspace(0,dvec(1),fix(abs(seg(i))*n)+2);
        dy = pvec(2)+linspace(0,dvec(2),fix(abs(seg(i))*n)+2);
        x = [x,dx];
        y = [y,dy];
        pvec = pvec+dvec;
    elseif type(i) == 'L'
        theta = pvec(3);
        dtheta = seg(i);
        if seg(i)==0
            continue
        end
        cenx = pvec(1)-rmin*sin(theta);
        ceny = pvec(2)+rmin*cos(theta);
        t = theta-pi/2+linspace(0,dtheta,fix(abs(seg(i))*n)+2);
        dx = cenx+rmin*cos(t);
        dy = ceny+rmin*sin(t);
        x = [x,dx];
        y = [y,dy];
        angle=[angle,t+pi/2];
        theta = theta+dtheta;
        pvec = [dx(end),dy(end),theta];
        dl = dtheta;
    elseif type(i) == 'R'
        theta = pvec(3);
        dtheta = -seg(i);
        if seg(i)==0
            continue
        end
        cenx = pvec(1)+rmin*sin(theta);
        ceny = pvec(2)-rmin*cos(theta);
        t = theta+pi/2+linspace(0,dtheta,fix(abs(seg(i))*n)+2);
        dx = cenx+rmin*cos(t);
        dy = ceny+rmin*sin(t);
        x = [x,dx];
        y = [y,dy];
        angle=[angle,t-pi/2];
        theta = theta+dtheta;
        pvec = [dx(end),dy(end),theta];
        dl = -dtheta;
    else
        % do nothing
    end
%     if dl > 0
%         plot(dx,dy,'b');
%     else
%         plot(dx,dy,'r');
%     end
%     hold on
end
route=[x' y' angle'];