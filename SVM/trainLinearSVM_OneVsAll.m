function [model] = trainLinearSVM_OneVsAll(matFeaturesTrain,matLabelsTrain, C)
%% Input order: (matData ,  matLabels)
% Data format [numSamples x featureDim]
% Label format [labels in first column]
% Labels are 1-based


disp('Make sure that order of arguments is: Features, Labels, C');


%% Class idx should start with 1,2,3,...
numClasses = max (matLabelsTrain(:,1));

if nargin < 3
    C = 100;
end

disp(['Train linear SVM one-vs-all with C = ' num2str(C) ...
    ' and numClasses = ' num2str(numClasses)]);

model = cell(numClasses,1);
for k = 1:numClasses
    disp(['Train class = ' num2str(k)])
    model{k} = svmtrain(double(matLabelsTrain(:,1) == k),matFeaturesTrain,['-c ' num2str(C) ' -b 1 -q']); % quiet version
end
