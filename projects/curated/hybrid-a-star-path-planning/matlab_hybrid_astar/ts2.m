clc
clear
close all
startPose = [1 1 pi/2]; % [meters, meters, radians]
goalPose = [7 1 pi/2];
min_r=5;
tic
[dis,route]=reeds_shepp_fun(startPose,goalPose,min_r);
toc
figure(1)
hold on
plot(startPose(1),startPose(2),'kx','LineWidth',2,'MarkerSize',10)
plot(goalPose(1),goalPose(2),'ko', 'LineWidth',2,'MarkerSize',10)
plot(route(:,1),route(:,2),'r-')

