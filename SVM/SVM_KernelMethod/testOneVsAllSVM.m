function [acc, ConfMat,Prob] = testOneVsAllSVM(model,KernelMatTest,matLabelsTest)

disp('Predict Actions using one-vs-all SVM')


%% Get number of classes from label matrix
numClasses = max (matLabelsTest(:,1));
numVids = size(matLabelsTest,1);
  
% Add column if sample indices are missing
if(size(matLabelsTest,2) == 1)
    disp('Column containing video indices is missing. Adding it automatically ...')
    matLabelsTest = cat(2,matLabelsTest,(1:numVids)');
end
 
disp(['numClasses = ' num2str(numClasses)])

Prob = zeros(size(matLabelsTest,1),numClasses);
for k=1:numClasses
    disp(['Test class = ' num2str(k)])
    [~,~,p] = svmpredict(double(matLabelsTest(:,1) == k), [matLabelsTest(:,2) KernelMatTest], model{k}, '-b 1');    
    Prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
end
  
%% Take maximum value as predicted class
[~,pred] = max(Prob,[],2);

%% Count how many predictions were correct
numCorrect = sum(pred == matLabelsTest(:,1));
numAll = numel(matLabelsTest(:,1));

acc = numCorrect ./ numAll *100;
disp('++++++++++++++++++++++++++++')
disp(['Accuracy: ' num2str(acc) ' (' num2str(numCorrect) '/' num2str(numAll) ')']);
disp('++++++++++++++++++++++++++++')

%% Calculate confusion matrix
ConfMat = confusionmat(matLabelsTest(:,1), pred);

%% Add labels and video number to matrix with positive estimates
Prob = [matLabelsTest Prob];