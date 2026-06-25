function [dis,route]=reeds_shepp_fun(startPose,goalPose,min_r)
x=goalPose(1)-startPose(1);
y=goalPose(2)-startPose(2);
goalPose(1)=x*cos(startPose(3))+y*sin(startPose(3));
goalPose(2)=-x*sin(startPose(3))+y*cos(startPose(3));
goalPose(3)=goalPose(3)-startPose(3);
if goalPose(3)>pi
    goalPose(3)=goalPose(3)-2*pi;
elseif goalPose(3)<=-pi
    goalPose(3)=goalPose(3)+2*pi;
end
path = FindRSPath(goalPose,min_r);
dis=min_r*path.totalLength;
route=getroute_fun(startPose,path,min_r);