% AutoEncoder for Image Classifcation
% requires the nueral net Toolbox in matlab

%loading the dataset
load('data.mat');

%displaying images
for i =1:20
	subplot(4,5,i);
	imshow(xTrainImages{i});
end

%setting the weights of a nueral net to default instead of random
rng('default')

%training in the first layer
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1,'MaxEpochs',400,'L2WeightRegularization',0.004,'SparsityRegularization',4,'SparsityProportion',0.15,'ScaleData',false);

%viewing the autoenc1
view(autoenc1)