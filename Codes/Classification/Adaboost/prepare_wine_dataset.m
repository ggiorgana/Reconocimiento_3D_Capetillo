function [A_train, A_labels_train, A_test, A_labels_test] = ... 
    prepare_wine_dataset()

A = importdata('datasets/wine.data');   % read dataset.
A = A(1:130,:);                         % get rid of class 3.
A_labels = A(:,1);                      % get labels from column 1.
A = A(:,2:14);                          % get rid of column 1 in set.
A_labels(60:131) = -1;                  % change class 2 by class -1.

% get 70% of dataset for training
A_train = [A(1:41,:); A(60:109,:)];
A_labels_train = [A_labels(1:41,1); A_labels(60:109,1)];

% get other 30% of dataset for testing
A_test = [A(42:59,:); A(110:130,:)];
A_labels_test = [A_labels(42:59,1); A_labels(110:130,1)];

end