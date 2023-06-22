classdef NeuralNetwork
    properties
        layers
        layerN
        lr
        guess
        err
        t
        mu
        s
    end

    methods
        function obj=init(obj)
            obj.layers={};
            obj.layerN=1;

        end
        function obj=addLayer(obj,layer)
            obj.layers{obj.layerN}=layer;
            obj.layerN=obj.layerN+1;
        end
        
        function obj=feedForward(obj,X)
            obj.layers{1}=obj.layers{1}.forward(X);
            for i=2:obj.layerN-1
                obj.layers{i}= obj.layers{i}.forward(obj.layers{i-1}.output);
            end
            obj.guess=obj.layers{obj.layerN-1}.output;
        end

        function obj=train(obj,X,d,lr,training_times)
            
            for t=1:training_times
                avgErr=0;
                for sample=1:size(X,2)
                    input=X(:,sample)';
                    obj=obj.feedForward(input);
                    desired=d(sample,:);
                    error=obj.guess-desired;
                    error=error';
                    avgErr=avgErr+sum(error.^2)/length(error);
                    for l=obj.layerN-1:-1:1
                        obj.layers{l}=obj.layers{l}.backwords(error,lr);
                        error=obj.layers{l}.input_error;
                    end
                    error=0;
                end
                obj.err{t}=avgErr/size(X,2);
            end
        end
        function plotErr(obj)
            plot(1:length(obj.err),cell2mat(obj.err))
        end
        function obj=getMuNs(obj,data)
                 obj.mu=mean(data);
                 obj.s=std(data);
        end

        

    end
    methods(Static)
        
        function data_out=shuffleData(data_in)

            [r c] = size(data_in);
            shuffledRow = randperm(r);
            data_out = data_in(shuffledRow, :);
        end
        
        function [training,validation,test]=splitData(data_in,train_perc,valid_perc,test_perc)
                [r c]=size(data_in);
                if(train_perc+valid_perc+test_perc~=100)
                    disp("Wrong percentages make sure it adds up to 100");
                    return
                end
                tr=r*train_perc/100;
                vr=r*valid_perc/100;
                ter=r*test_perc/100;
                
                training=data_in(1:floor(tr),:);
                validation=data_in(floor(tr)+1:floor(tr)+floor(vr),:);
                test=data_in(floor(tr)+1+floor(vr)+1:end,:);

        end
        function data_out=normalizeData(data_in,mu,s)
            data_out=(data_in-mu)./s;
        end
    
    end


end




