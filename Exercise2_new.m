
clc; clear all; close all;
a=[3,4,2,3];

X=rand(3,5);

out=zeros(3,5);

for j=1:5
    out(1,j)=X(1,j)^2+X(2,j)+X(3,j)^3;
    out(2,j)=X(1,j)^4+X(2,j)^2+X(3,j)^3;
    out(3,j)=X(1,j)+X(2,j)+X(3,j)^2;
end

NN=createNN(a,@sig);

for l=1:1000
for k=1:5
    NN.W=train(NN,X(:,k),out(:,k),0.0000001,@sig,@dsigmoid,1000);
end
end
for k=1:5
[Vs,Ys]=FeedForward(NN,X(:,k));
Ys{5}
end

out

function s=sig(x)
 s=1./(exp(-x)+1);
end
function ds=dsigmoid(x)
    ds=sig(x).*(1-sig(x));
end

function NN=createNN(layers,activation_func)
    number_of_layers=length(layers);
    W={};
    for i=1:number_of_layers-1
        %Create Weights
        W{i}=(rand(layers(i+1),layers(i))*2)-1;

           

    end

NN.W=W
NN.layers=layers
NN.number_of_layers=number_of_layers-1
NN.activation_func=activation_func;
end

function [Vs,Ys]=FeedForward(NN,X)
        number_of_samples=size(X,2);
        V={};
        Y={};
        for i=2:NN.number_of_layers

                V{i-1}=zeros(number_of_samples,NN.layers(i));
                Y{i-1}=zeros(number_of_samples,NN.layers(i));
        end
        
        V{1}=NN.W{1}*X;
        Y{1}=NN.activation_func(V{1});
        for i=2:NN.number_of_layers
            V{i}=NN.W{i}*V{i-1};
            Y{i}=NN.activation_func(V{i});
        end
Ys=Y;
Vs=V;

end

function [W]=train(NN,X,d,lr,acti,dacti,iterations)
   % for iter=1:iterations   
        [Vs,Ys]=FeedForward(NN,X);
       
        for l=NN.number_of_layers:-1:1
            if l==NN.number_of_layers
             error{l}=Ys{l}-d;
            else
             error{l}=error{l+1}*NN.W{l};
            end
        end
            
        
            for i=1:NN.number_of_layers-1
                %NN.W{i}=NN.W{i}-lr.*dw{i};

                NN.W{i}=NN.W{i}-lr.*dw{i};
            end
%    end

W=NN.W;
end

