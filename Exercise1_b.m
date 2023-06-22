close all; clear all; clc;
data=load('.\datasets\dataLinearSeparability.mat');
N=length(data.X0);
X0=zeros(N,length(cell2mat(data.X0(1))));
X1=zeros(N,length(cell2mat(data.X1(1))));
for i=1:N
    X0(i,:)=cell2mat(data.X0(i))';
    X1(i,:)=cell2mat(data.X1(i))';
end
X0=[ones(N,1),X0]';
X1=[ones(N,1),X1]';
d0=(ones(N,1));
d1=(-ones(N,1));
W=zeros(8,1);
inputD=[X0,X1];
a=0.0001;
ds=[d0;d1];
for i=1:5
    y=sign(W'*inputD);
    W= W + a*(inputD*(ds-y'));

end

N0=length(X0);
N1=length(X1);
x1=W'*X0;
y1=W'*X1;
maxX=max(-x1);
maxY=max(y1);
maxBoth=max(maxX,maxY);
step=(2*maxBoth)/N0;
t=-maxBoth:step:maxBoth;
plot(x1,t(1:10000),"*",'Color',"red");
hold on
plot(y1,t(1:10000),"*",'Color',"blue");

legend("X","Y");
