function label_free_fitctree(trainCsv, testCsv)
% LABEL_FREE_FITCTREE  Train a Gini decision tree and a bagged forest on the
% synthetic SEALS spectra exported by scarlett.label_free.export_csv, and
% print test accuracies. Requires the Statistics and Machine Learning Toolbox.
%
%   matlab -batch "addpath('matlab'); label_free_fitctree('train.csv','test.csv')"
%
% Last column of each CSV is the class label; the rest are log10 intensities.

tr = readmatrix(trainCsv);  te = readmatrix(testCsv);
Xtr = tr(:, 1:end-1);  ytr = tr(:, end);
Xte = te(:, 1:end-1);  yte = te(:, end);

rng(0);
tree   = fitctree(Xtr, ytr, 'SplitCriterion', 'gdi', 'MaxNumSplits', 20);   % gdi = Gini's diversity index
forest = TreeBagger(60, Xtr, ytr, 'Method', 'classification', 'MinLeafSize', 2);

accTree   = mean(predict(tree, Xte) == yte);
accForest = mean(str2double(predict(forest, Xte)) == yte);
fprintf('MATLAB fitctree (gini)   test accuracy = %.4f\n', accTree);
fprintf('MATLAB TreeBagger (60)   test accuracy = %.4f\n', accForest);
end
