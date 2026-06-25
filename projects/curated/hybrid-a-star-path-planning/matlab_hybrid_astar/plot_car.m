function [h1,h2]=plot_car(coo_car,size_car)
h1=plot(coo_car(1),coo_car(2), 'o','markersize',5,'markerfacecolor','y','MarkerEdgeColor', 'k');
alphy1=atan((size_car(2)/2)/size_car(3));%右上
alphy2=-alphy1;%右下
alphy3=pi+atan((size_car(2)/2)/(size_car(1)-size_car(3)));%左下
alphy4=pi-atan((size_car(2)/2)/(size_car(1)-size_car(3)));%左上
alphy1=alphy1+coo_car(3);
alphy2=alphy2+coo_car(3);
alphy3=alphy3+coo_car(3);
alphy4=alphy4+coo_car(3);
l1=((size_car(2)/2)^2+size_car(3)^2)^0.5;
l2=((size_car(2)/2)^2+(size_car(1)-size_car(3))^2)^0.5;
x1(1)=coo_car(1)+l1*cos(alphy1);
y1(1)=coo_car(2)+l1*sin(alphy1);
x1(2)=coo_car(1)+l1*cos(alphy2);
y1(2)=coo_car(2)+l1*sin(alphy2);
x1(3)=coo_car(1)+l2*cos(alphy3);
y1(3)=coo_car(2)+l2*sin(alphy3);
x1(4)=coo_car(1)+l2*cos(alphy4);
y1(4)=coo_car(2)+l2*sin(alphy4);
h2=fill(x1,y1,'k');
set(h2,'facealpha',0.2)

