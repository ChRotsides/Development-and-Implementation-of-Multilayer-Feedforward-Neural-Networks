clear all;clc; close all;

trainingSet1=load("datasets\trainingSet1.dat");

X=trainingSet1(:,1:2);
y=trainingSet1(:,3);


nn=NeuralNetwork;
nn=nn.init();


nn=nn.addLayer(Layer().init(2,4));
nn=nn.addLayer(activationLayer().init(@activationLayer.sig,@activationLayer.dsigmoid));
nn=nn.addLayer(Layer().init(4,7));
nn=nn.addLayer(activationLayer().init(@tanh,@activationLayer.tanhP));
nn=nn.addLayer(Layer().init(7,5));
nn=nn.addLayer(Layer().init(5,1));


%X=X-(mean(X,2));
for i=1:length(X)

 %   X(i,:)=X(i,:)/std(X(i,:));
end
X=X';
for i=1:length(y)
    if(y(i)==0)
        y(i)=-1;
    end
end
guesses=[];
%out=Layer.sig(out);
%nn=nn.feedForward(X(:,1),0)
nn=nn.train(X,y,0.008,10);
for i=1:length(X)
nn=nn.feedForward(X(:,i)');
guesses(i)=nn.guess;
end

save("trainedNetwork"+nn.layerN+"Layers_OnlyNewFunc"+datestr(now),'nn');

