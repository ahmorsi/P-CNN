function model = trainOneVsAllSVM(KernelMatTrain,matLabelsTrain,C)
% matLabelsTrain must have 2 columns [numSamples x Label,SampleNumber]
% (sampleNumber is a 1 based index)
% Labels must be 1 indexed (see " == k" check below)

  disp('Train SVM model')
  
  numVids = size(matLabelsTrain,1);
  numClasses = max (matLabelsTrain(:,1));
  disp(['numClasses = ' num2str(numClasses)])
  disp(['C = ' num2str(C)])
  
  model = cell(numClasses,1);
  
  % Add column if sample indices are missing
  if(size(matLabelsTrain,2) == 1)
        disp('Column containing video indices is missing. Adding it automatically ...')
        matLabelsTrain = cat(2,matLabelsTrain,(1:numVids)');
  end
  
  for k=1:numClasses
    disp(['Train class = ' num2str(k)])
    model{k} = svmtrain(double(matLabelsTrain(:,1) == k), [matLabelsTrain(:,2) KernelMatTrain],['-t 4 -c ' num2str(C) ' -b 1 -q']); % quiet version
  end
  
  
end



