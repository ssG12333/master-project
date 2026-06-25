function path = FindRSPath(goalPose,rmin)
    x = goalPose(1)/rmin;
    y = goalPose(2)/rmin;
    phi = goalPose(3);
    % 遍历5种方法到达目标点，然后选取路径最短的一条
    [isok1,path1] = CSC(x,y,phi);
    [isok2,path2] = CCC(x,y,phi);
    [isok3,path3] = CCCC(x,y,phi);
    [isok4,path4] = CCSC(x,y,phi);
    [isok5,path5] = CCSCC(x,y,phi);
    isoks = [isok1, isok2, isok3, isok4, isok5];
    paths = {path1, path2, path3, path4, path5};
    Lmin = inf;
    % 找出5条路径最短的曲线
    for i = 1:5
        if isoks(i) == true
            elem = paths{i};
            if Lmin > elem.totalLength
                Lmin = elem.totalLength;
                path = elem;
            end
        end
    end
%     PlotPath(path,veh);
end













 