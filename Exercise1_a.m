
clear all; close all; clc;
clear('model');
X = load('.\datasets\dataLinearSeparability.mat');

%model.obj = [1 2 3];

X0=X.X0;
X1=X.X1;
N0=length(X0);
N1=length(X1);
x=zeros(length(X0),length(cell2mat(X0(1))));
y=zeros(length(X1),length(cell2mat(X1(1))));
for i=1:length(X0)
    x(i,:)=cell2mat(X0(i))';
end
for i=1:length(X0)
    y(i,:)=cell2mat(X1(i))';
end
model.A = sparse([-x;y]);
%model.modelsense = 'Max';
model.sense(1:N0) ='>';
model.sense(N0+1:N0+N1)='>';
model.rhs(1:N0) =10^-3;
model.rhs(N0+1:N0+N1)=10^-3;
model.lb(1:7)=-inf;
model.ub(1:7)=inf;
result = gurobi(model);
res=result.x;
   for i=1:N0
           if(~(x(i,:)*res <-10^-3))
                disp("X:"+i);
           end
   end

    for i=1:N1
        if(~(y(i,:)*res>10^-3-0.0001))
            disp("Y:"+i);
        end
    end

x1=x*res;
y1=y*res;
maxX=max(-x1);
maxY=max(y1);
maxBoth=max(maxX,maxY);
step=(2*maxBoth)/N0;
t=-maxBoth:step:maxBoth;
plot(x1,t(1:10000),"*",'Color',"red");
hold on
plot(y1,t(1:10000),"*",'Color',"blue");

legend("X","Y");

result.x