clc
clear
close all
row=10;
col=10;
sign=zeros(row,col);
sign(4:5,4:5)=1;%设置障碍
sign(8,5)=1;
startPose = [1.6 0.8 pi/2]; % [meters, meters, radians]
goalPose = [9.1 8.8 pi/3];%终点
min_r=1;%最小转弯半径
safe_dis=0.5;%与障碍物的安全距离
step=0.5;%步长
P3=0.01;%角度惩罚系数
ob_coo=[];
figure(1)%画障碍图
hold on
axis equal
for i=1:row
    for j=1:col
        if sign(i,j)==1
            y=[i-1,i-1,i,i];
            x=[j-1,j,j,j-1];
            h=fill(x,y,'k');
            set(h,'facealpha',1)
            ob_coo=[ob_coo;[j-0.5,i-0.5]];
        end
        %         %s=(num2str((i-1)*col+j));
        %         s=(['(',num2str(i),',',num2str(j),')']);
        %         text(j-0.95,i-0.5,s,'fontsize',8)
    end
end
axis([0 col 0 row])%限制图的边界
for i=1:row
    plot([0 col],[i i],'k-');
end
for i=1:col
    plot([i i],[0 row],'k-');%画网格线
end
plot(startPose(1),startPose(2), 'p','markersize', 10,'markerfacecolor','b','MarkerEdgeColor', 'm')
plot(goalPose(1),goalPose(2),'o','markersize', 10,'markerfacecolor','g','MarkerEdgeColor', 'c')
% set(gca,'YDir','reverse');

opened=[startPose 0 0 0 0 1 0];%x,y,sita,方向、g值，h值，f值，二叉树中的位置，父代编号
dis_astar=Astar_fun(fliplr(ceil(startPose(1:2))),fliplr(ceil(goalPose(1:2))),sign);
[dis_rs,~]=reeds_shepp_fun(startPose,goalPose,min_r);
opened(1,6)=max(dis_astar,dis_rs);
opened(1,7)=opened(1,5)+opened(1,6);
now_point=opened;
nodeIndex = 1;%节点编码
nodeArray(nodeIndex).x=now_point(1);
nodeArray(nodeIndex).y=now_point(2);
nodeArray(nodeIndex).sita=now_point(3);
nodeArray(nodeIndex).D=now_point(4);
nodeArray(nodeIndex).g=now_point(5);
nodeArray(nodeIndex).h=now_point(6);
nodeArray(nodeIndex).f=now_point(7);
nodeArray(nodeIndex).ind=now_point(8);
nodeArray(nodeIndex).parent=now_point(9);
nodeArray(nodeIndex).route=[];
nodeIndex = nodeIndex + 1;
[min_num,index] = min(opened(:,7));
while ceil(now_point(1))~=ceil(goalPose(1))||ceil(now_point(2))~=ceil(goalPose(2))   %结束条件(也就是终点被放入到)，
    opened(index,:) = [];     %从开放列表中删除
    for i=1:3
        [isok,x,y,sita,route]=find_route_fun(now_point,i,step,min_r,safe_dis,ob_coo);
        if x<=0||x>col||y<=0||y>row
            isok=1;
        end
        if isok==0
            temp=[x,y,sita,0,0,0,0,nodeIndex,now_point(8)];
            if i>3
                temp(4)=1;
            end
            temp(5)=now_point(5)+step+sum(abs(temp(3)-now_point(3)))*P3;
            dis_astar=Astar_fun(fliplr(ceil(temp(1:2))),fliplr(ceil(goalPose(1:2))),sign);
            [dis_rs,~]=reeds_shepp_fun(temp(1:3),goalPose,min_r);
            temp(6)=max(dis_astar,dis_rs);
            temp(7)=temp(5)+temp(6);
            opened=[opened;temp];
            nodeArray(nodeIndex).x=temp(1);
            nodeArray(nodeIndex).y=temp(2);
            nodeArray(nodeIndex).sita=temp(3);
            nodeArray(nodeIndex).D=temp(4);
            nodeArray(nodeIndex).g=temp(5);
            nodeArray(nodeIndex).h=temp(6);
            nodeArray(nodeIndex).f=temp(7);
            nodeArray(nodeIndex).ind=temp(8);
            nodeArray(nodeIndex).parent=temp(9);
            nodeArray(nodeIndex).route=route;
            nodeIndex = nodeIndex + 1;
            if i<=3
                plot(route(:,1),route(:,2),'b-')
            else
                plot(route(:,1),route(:,2),'g-')
            end
            drawnow
            plot(route(end,1),route(end,2), 'o','markersize', 2,'markerfacecolor','r','MarkerEdgeColor', 'm')
        end
    end
    [min_num,index] = min(opened(:,7));
    now_point=opened(index,:);
    if norm(now_point(1:2)-goalPose(1:2))<2*step
        break
    end
end
node_temp=nodeArray(now_point(8));
index=now_point(8);
while node_temp.parent~=0
    node_temp=nodeArray(node_temp.parent);
    index=[index node_temp.ind];
end
index=fliplr(index);
route_all=startPose;
for i=1:length(index)
    route=nodeArray(index(i)).route;
    route_all=[route_all;route(2:end,:)];
end
[~,route]=reeds_shepp_fun(now_point(1:3),goalPose,min_r);
route_all=[route_all;route(2:end,:)];
size_car=[1 0.6 0.4];
[h1,h2]=plot_car(route_all(1,:),size_car);
dis_all=0;
for i=2:size(route_all,1)
    dis_all=dis_all+norm(route_all(i,1:2)-route_all(i-1,1:2));
    r_lin=route_all(i,:);
    delete(h1)
    delete(h2)
    plot(route_all(i-1:i,1),route_all(i-1:i,2),'r-','LineWidth',2)
    [h1,h2]=plot_car(r_lin,size_car);
    drawnow;
    %pause(0.03);
end
disp(['总距离：',num2str(dis_all)])
% [h1,h2]=plot_car(route(end,:),size_car);
% plot(route_all(:,1),route_all(:,2),'k-','LineWidth',2)



