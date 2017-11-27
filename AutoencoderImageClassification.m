% AutoEncoder for Image Classifcation
% requires the nueral net Toolbox in matlab

%loading the dataset
load('data.mat');

%{displaying images
for i =1:20
	subplot(4,5,i);
	imshow(xTrainImages{i});
end
%}

%setting the weights of a nueral net to default instead of random
rng('default')

%training in the first layer
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1,'MaxEpochs',400,'L2WeightRegularization',0.004,'SparsityRegularization',4,'SparsityProportion',0.15,'ScaleData',false);

%viewing the autoenc1
view(autoenc1)

%{plotting the weights 
figure()
plotWeights(autoenc1);
%}

%getting the features from the first encoder
feat1 = encode(autoenc1,xTrainImages);


% training the 2nd autoencoder
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2,'MaxEpochs',100,'L2WeightRegularization',0.002,'SparsityRegularization',4,'SparsityProportion',0.1,'ScaleData',false);

%viewing the autoenc2
view(autoenc2)

%getting the features of the second layer 
feat2 = encode(autoenc2,feat1);

%training the final softmax layer
softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);

%viewing the softmax layer
view(softnet)

%stacking to form a deepnet
deepnet = stack(autoenc1,autoenc2,softnet);

%viewing the deepnet
view(deepnet)

%modifying the test images into vectors
inputSize = 28*28;

%loading the testImages 
[xTestImages,tTest] = digitTestCellArrayData;
