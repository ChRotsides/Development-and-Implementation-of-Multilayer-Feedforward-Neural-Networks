nn1=NeuralNetwork;

nn1=nn1.init();
data=load("datasets\trainingSet1.dat");
data=NeuralNetwork.shuffleData(data);
[trainingData ,validation ,test]=NeuralNetwork.splitData(data,70,30,0);

Xval=validation(:,1:2);
Yval=validation(:,3);

Xtrain=trainingData(:,1:2);
Ytrain=trainingData(:,3);

nn1=load("nn1.mat").nn1;
nn2=load("nn2.mat").nn2;
nn3=load("nn3.mat").nn3;

Xtrain=NeuralNetwork.normalizeData(Xtrain,nn1.mu,nn1.s);
Xval=NeuralNetwork.normalizeData(Xval,nn1.mu,nn1.s);
mean(Xtrain)
std(Xtrain)
disp("Mean of Train Data: "+mean(Xtrain));
disp("sigma of Train Data: "+std(Xtrain));



for i=1:length(Yval)
    if(Yval(i)==0)
        Yval(i)=-1;
    end
end

for i=1:length(Ytrain)
    if(Ytrain(i)==0)
        Ytrain(i)=-1;
    end
end

%nn1=nn1.train(Xtrain',Ytrain,0.001,200);
%nn2=nn2.train(Xtrain',Ytrain,0.01,200);
%nn3=nn3.train(Xtrain',Ytrain,0.001,200);
nn1Correct=0;
nn2Correct=0;
nn3Correct=0;
for i=1:length(Yval)
    if(nn1.feedForward(Xval(i)').guess==Yval(i))
        nn1Correct=nn1Correct+1;
    end
    if(nn2.feedForward(Xval(i)').guess==Yval(i))
        nn2Correct=nn2Correct+1;
    end
    if(nn3.feedForward(Xval(i)').guess==Yval(i))
        nn3Correct=nn3Correct+1;
    end
end

nn1Accuracy=nn1Correct/length(Yval);
nn2Accuracy=nn2Correct/length(Yval);
nn3Accuracy=nn3Correct/length(Yval);
disp("NN1 Accuracy:"+ nn1Accuracy);
disp("NN2 Accuracy:"+ nn2Accuracy);
disp("NN3 Accuracy:"+ nn3Accuracy);
hold on
nn1.plotErr();
nn2.plotErr();
nn3.plotErr();
legend("nn1","nn2","nn3");


