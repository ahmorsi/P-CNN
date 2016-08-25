function [acc, ConfMat,Prob,Pred] = testLinearSVM_OneVsAll(matFeaturesTest,matLabelsTest,model)


%% predict labels
%% In matLabelsTest: Class idx should start with 1,2,3,...
disp('Predict Actions using one-vs-all SVM')

numClasses = max (matLabelsTest(:,1));

disp(['numClasses = ' num2str(numClasses)])
%% Get number of classes from label matrix
Prob = zeros(size(matLabelsTest,1),numClasses);
  for k=1:numClasses
    disp(['Test class = ' num2str(k)])
    [~,~,p] = svmpredict(double(matLabelsTest(:,1) == k), matFeaturesTest, model{k}, '-b 1');
    Prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
  end
  
  %% Take maximum value as predicted class
  [~,Pred] = max(Prob,[],2);
  
  %% Count how many predictions were correct
  numCorrect = sum(Pred == matLabelsTest(:,1));
  numAll = numel(matLabelsTest(:,1));
  
  acc = numCorrect ./ numAll *100;
  disp('++++++++++++++++++++++++++++')
  disp(['Accuracy: ' num2str(acc) ' (' num2str(numCorrect) '/' num2str(numAll) ')']);
  disp('++++++++++++++++++++++++++++')
  
  %% Calculate confusion matrix
  ConfMat = confusionmat(matLabelsTest(:,1), Pred);
  
  %% Add labels and video number to matrix with positive estimates
  % Prob = [pred Prob];
  
