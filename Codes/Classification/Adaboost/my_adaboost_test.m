function [H acc] = my_adaboost_test(data, labels, model_fname)
% data = NxM matrix containing the training data. N = the number of
%training data, while M is the number of features of each data.
% labels = Nx1 vector containing the label of each training data.
    
    % Read the stored model.
    fileID = fopen(model_fname,'rt');
    num_weak_learners = cell2mat(textscan(fileID, '%*s %d'));
    %num_weak_learners = 34;
    fgets(fileID);
    formatSpec = '%f %d %d %f';
    sizeWeakLearners = [4 num_weak_learners];
    temp = fscanf(fileID,formatSpec,sizeWeakLearners)';
    fclose(fileID);

    % Create variables to run adaboost.    
    a = struct('threshold',0,'ineq',0,'feat_idx',0,'alpha',0);
    weak_learner = repmat(a,num_weak_learners,1);
    
    for i = 1:num_weak_learners
        weak_learner(i).threshold = temp(i,1);
        weak_learner(i).ineq = temp(i,2);
        weak_learner(i).feat_idx = temp(i,3);
        weak_learner(i).alpha = temp(i,4);
    end
    
    %For each test instance
    for i=1:size(data,1)
        %Evaluate all weak learners.
        sum = 0;
        for t=1:num_weak_learners
            if(data(i,weak_learner(t).feat_idx) < weak_learner(t).threshold)
                hyp = 1;
            else
                hyp = -1;
            end

            hyp = hyp*weak_learner(t).ineq;

            sum = sum + weak_learner(t).alpha*hyp;
        end
        
        if(sum>=0)
            H(i) = 1;
        else
            H(i) = -1;
        end
    end
        
    if(isrow(H))
        H = H';
    end
%     labels';
    
    acc = 0;
    for i=1:size(data,1)
        if(H(i) == labels(i))
            acc = acc + 1;
        else
            i;
            H(i);
            labels(i);
        end
    end
    
    acc;
    acc = acc/size(data,1);
    
end