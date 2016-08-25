splits = 3;
PcnnOutput = '/media/data/amorsy/Full_JHMDB/cnnfeatures';

classes = dir(PcnnOutput);
classes = {classes.name};
classes=classes(~ismember(classes,{'.','..'}));

for splitIdx=1:splits
    %create Split Struct(trainFeatures,trainLabels,testFeatures,testLabels)
    splitData = struct('trainData',[],'testData',[],'trainLabels',[],'testLabels',[]);
    for ci=1:length(classes)
        className = classes{ci};
        trainFeatures = load(sprintf('%s/%s/p-cnn_features_split%d/Xn_train.mat',PcnnOutput,className,splitIdx));
        testFeatures = load(sprintf('%s/%s/p-cnn_features_split%d/Xn_test.mat',PcnnOutput,className,splitIdx));
        
        splitData.trainData = [splitData.trainData;trainFeatures.Xn_train'];
        splitData.testData = [splitData.testData;testFeatures.Xn_test'];

	trainLabels = zeros(size(trainFeatures.Xn_train,2),1);
    testLabels = zeros(size(testFeatures.Xn_test,2),1);
	trainLabels(:) = ci;
	testLabels(:) = ci;
	
	splitData.trainLabels = [splitData.trainLabels;trainLabels];
    splitData.testLabels = [splitData.testLabels;testLabels];
    end
    data.kernelTrainData = splitData.trainData*splitData.trainData';
    data.kernelTestData = splitData.testData*splitData.trainData';	
    save(sprintf('%s/splitData%d.mat',PcnnOutput,splitIdx),'-struct','splitData');
end
