% Assuming XTrain, YTrain are your training features and labels

% Set up partition for cross-validation
c = cvpartition(YTrain, 'KFold', 10);

% Set up the parameter grid to search
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus');

% Train the SVM model with kernel and box constraint tuning
svmModel = fitcsvm(XTrain, YTrain, 'KernelFunction','rbf', ...
    'OptimizeHyperparameters',{'BoxConstraint', 'KernelScale'}, ...
    'HyperparameterOptimizationOptions',opts);

% Predict using the optimized SVM model
YPred = predict(svmModel, XTest);

% Calculate accuracy and other metrics
accuracy = sum(YPred == YTest) / length(YTest);
confMat = confusionmat(YTest, YPred);
fprintf('Accuracy of optimized SVM model: %.2f%%\n', accuracy * 100);

% Display the confusion matrix
disp('Confusion Matrix:');
disp(confMat);
