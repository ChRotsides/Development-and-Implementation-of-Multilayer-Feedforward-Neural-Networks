clear all;close all;clc;
nn1=NeuralNetwork;

nn1=nn1.init();
data=load("datasets\trainingSet2.dat");
dataTest=load("datasets\testingSet2.dat");
data=NeuralNetwork.shuffleData(data);
[trainingData ,validation ,test]=NeuralNetwork.splitData(data,70,30,0);

Xval=validation(:,1:2);
Yval=validation(:,3);

Xtrain=trainingData(:,1:2);
Ytrain=trainingData(:,3);

Xtest=dataTest(:,1:2);
Ytest=dataTest(:,3);
Xtrain=[Xtrain;Xval];
Ytrain=[Ytrain;Yval];

nn1=nn1.getMuNs(Xtrain);


Xtrain=NeuralNetwork.normalizeData(Xtrain,nn1.mu,nn1.s);
Xtest=NeuralNetwork.normalizeData(Xtest,nn1.mu,nn1.s);
mean(Xtrain)
std(Xtrain)
disp("Mean of Train Data: "+mean(Xtrain));
disp("sigma of Train Data: "+std(Xtrain));


nn1=nn1.addLayer(Layer().init(2,5));
nn1=nn1.addLayer(activationLayer().init(@activationLayer.sig,@activationLayer.dsigmoid));
nn1=nn1.addLayer(Layer().init(5,10));
nn1=nn1.addLayer(activationLayer().init(@tanh,@activationLayer.tanhP));
nn1=nn1.addLayer(Layer().init(10,1));
nn1=nn1.addLayer(activationLayer().init(@activationLayer.sig,@activationLayer.dsigmoid));



nn1=nn1.train(Xtrain',Ytrain,0.001,200);

nn1Correct=0;

for i=1:length(Ytest)
    if(round(nn1.feedForward(Xtest(i,:)).guess)==Ytest(i))
        nn1Correct=nn1Correct+1;
    end
end

nn1Accuracy=nn1Correct/length(Ytest);


nn1.plotErr();



