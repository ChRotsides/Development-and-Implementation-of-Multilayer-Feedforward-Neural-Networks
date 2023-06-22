
clc; clear all; close all;
a=[3,2,6,4,3];



NN=createNN(a,@sig)

X=rand(3,5);

out=zeros(3,5);

for j=1:5
    out(1,j)=X(1,j)^2+X(2,j)+X(3,j)^3;
    out(2,j)=X(1,j)^4+X(2,j)^2+X(3,j)^3;
    out(3,j)=X(1,j)+X(2,j)+X(3,j)^2;
end

NN.train(NN.W,X,out,@dsigmoid)
NN.feedforward(X,NN.W)



function s=sig(x)
 s=1./(exp(-x)+1);
end
function ds=dsigmoid(x)
    ds=sig(x).*(1-sig(x));
end
function NN=createNN(layers,activation_function)
    
    N=length(layers);

    W={};
    V={};
    Y={};
    for i=1:N-1
        W{i}=ones(layers(i+1),layers(i));
        V{i}=zeros(layers(i+1));
        Y{i}=zeros(layers(i+1),1);
    end
    act=activation_function;
    
    function [result,Vs]=feedforward(X,W)
                V{1}=W{1}*X;
                Y{1}=act(V{1});
                for i=2:length(W)
                    V{i}=W{i}*V{i-1};
                    Y{i}=act(V{i});
                end

                result=Y;
                Vs=V;
           end

    function nW=train(W,input,ds,dactivation_func)
            %feedforward to get guess
            [ys,vs]=feedforward(input,W);
            %as=zeros(4,1);
            %sum the guess for multiple inputs
        
            layers_size=length(W);
            %first from last
            errors={};
        %   dact=dactivation_func(ys{length(ys)})
         %  dw=error.*dact*vs{length(vs)}';

            for i=layers_size:-1:2
                if i==layers_size
                   error=(sum(ds,2)-ys{length(ys)});
                    
                   errors{i}= error./N;
                else
                %W{i}=W{i}-0.001*dw
                temp=errors{i+1}'*W{i};
                errors{i}=temp.*dactivation_func(vs{i+1});
                end

            end


            

    end


NN.feedforward=@feedforward;
NN.W=W;
NN.train=@train;


end



