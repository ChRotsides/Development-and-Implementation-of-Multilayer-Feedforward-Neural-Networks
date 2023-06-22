nn1=NeuralNetwork;

nn1=nn1.init();
data=load("datasets\trainingSet1.dat");
%data=NeuralNetwork.shuffleData(data);
[trainingData ,validation ,test]=NeuralNetwork.splitData(data,70,30,0);

Xval=validation(:,1:2);
Yval=validation(:,3);

Xtrain=trainingData(:,1:2);
Ytrain=trainingData(:,3);

nn1=nn1.getMuNs(Xtrain);
nn2=nn1.getMuNs(Xtrain);
nn3=nn1.getMuNs(Xtrain);

Xtrain=NeuralNetwork.normalizeData(Xtrain,nn1.mu,nn1.s);
Xval=NeuralNetwork.normalizeData(Xval,nn1.mu,nn1.s);
mean(Xtrain)
std(Xtrain)
disp("Mean of Train Data: "+mean(Xtrain));
disp("sigma of Train Data: "+std(Xtrain));


nn1=nn1.addLayer(Layer().init(2,10));
nn1=nn1.addLayer(activationLayer().init(@activationLayer.sig,@activationLayer.dsigmoid));
nn1=nn1.addLayer(Layer().init(10,20));
nn1=nn1.addLayer(activationLayer().init(@tanh,@activationLayer.tanhP));
nn1=nn1.addLayer(Layer().init(20,1));
nn1=nn1.addLayer(activationLayer().init(@sign,@activationLayer.signP));

nn2=nn2.addLayer(Layer().init(2,10));
nn2=nn2.addLayer(activationLayer().init(@activationLayer.sig,@activationLayer.dsigmoid));
nn2=nn2.addLayer(Layer().init(10,20));
nn2=nn2.addLayer(activationLayer().init(@tanh,@activationLayer.tanhP));
nn2=nn2.addLayer(Layer().init(20,1));
nn2=nn2.addLayer(activationLayer().init(@sign,@activationLayer.signP));

nn3=nn3.addLayer(Layer().init(2,5));
nn3=nn3.addLayer(activationLayer().init(@activationLayer.sig,@activationLayer.dsigmoid));
nn3=nn3.addLayer(Layer().init(5,10));
nn3=nn3.addLayer(activationLayer().init(@tanh,@activationLayer.tanhP));
nn3=nn3.addLayer(Layer().init(10,1));
nn3=nn3.addLayer(activationLayer().init(@sign,@activationLayer.signP));

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

nn1=nn1.train(Xtrain',Ytrain,0.001,100);
nn2=nn2.train(Xtrain',Ytrain,0.01,100);
nn3=nn3.train(Xtrain',Ytrain,0.001,100);
nn1Correct=0;
nn2Correct=0;
nn3Correct=0;
for i=1:length(Yval)
    if(sign(nn1.feedForward(Xval(i,:)).guess)==Yval(i))
        nn1Correct=nn1Correct+1;
    end
    if(sign(nn2.feedForward(Xval(i,:)).guess)==Yval(i))
        nn2Correct=nn2Correct+1;
    end
    if(sign(nn3.feedForward(Xval(i,:)).guess)==Yval(i))
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
hold on
nn2.plotErr();
hold on
nn3.plotErr();

legend("nn1","nn2","nn3");


