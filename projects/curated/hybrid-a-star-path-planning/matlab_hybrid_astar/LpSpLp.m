% formula 8.1
function [isok,t,u,v] = LpSpLp(x,y,phi)
    [t,u] = cart2pol(x-sin(phi),y-1+cos(phi)); % 将笛卡尔坐标转换为极坐标,返回theta和rho,论文返回的是[u,t],是因为cart2pol函数返回的值的顺序不同导致与原文不同，变量代表的含义还是一样，t代表弧度，u代表直行的距离
    if t >= 0 % 必须是左转,t>=0代表左转
        v = mod2pi(phi-t);
        if v >= 0 % 符号代表前进和后退
            isok = true;
            return
        end
    end
    isok = false;
    t = 0;
    u = 0;
    v = 0;
end