function my_adaboost_train(data, labels, num_epochs, model_fname)
% data = NxM matrix containing the training data. N = the number of
%training data, while M is the number of features of each data.
% labels = Nx1 vector containing the label of each training data.

    debug = 'no';

    % Create the file where the model will be stored.
    fileID = fopen(model_fname,'wt');
    fprintf(fileID, 'Num_weak_learners= %d\n\n', num_epochs);
    header = '%s\t\t %s\t\t %s\t %s\n';
    fprintf(fileID, header, '   th' , 'ineq', 'feat_idx', '  alpha');
    fclose(fileID);
    
    % Create the variables necessary to run adaboost.
    N = size(data,1); % number of training data.
    M = size(data,2); % number of features of each data.
    D = ones(1,N)/N;  % Create and initialize the distribution of weights.
    dec_stump = zeros(1,M);
    error = zeros(1,M);     % error of each decision stump.
    hyp = zeros(N,M);
    ineq = ones(1,M);
    weak_learner = {};

    %Find the M decision stumps (one per feature).
    for j=1:M
        K = 0;
        L = 0;
        mean_pos = 0;   % mean value of the positive training examples (1 feature).
        mean_neg = 0;   % mean value of the negative training examples (1 feature).
        for i=1:N
            if(labels(i) == 1)
                mean_pos = mean_pos + data(i,j);
                K = K+1;
            elseif(labels(i) == -1)
                mean_neg = mean_neg + data(i,j);
                L = L+1;
            end       
        end
        dec_stump(j) = (mean_pos/K + mean_neg/L)/2;
    end
    
    if(strcmp(debug,'yes') == 1)
        dec_stump
    end

    for t = 1:num_epochs 
        %Find the error of each decision stump.
        %Find the hypothesis of each decision stump.
        for j=1:M
            for i=1:N
                if(data(i,j) < dec_stump(j));
                    hyp(i,j) = 1;
                else
                    hyp(i,j) = -1;
                end
            end
        end                
        
        if(strcmp(debug,'yes') == 1)
            t
            hyp
            labels'
        end

        % score the M hypothesis.
        error = error*0; % initialize error to zero.   
        for j=1:M
            for i=1:N
                if(hyp(i,j) ~= labels(i))
                    error(j) = error(j) + D(i);
                end
            end
        end
        
        if(strcmp(debug,'yes') == 1)
            error
        end

        % Find the hypothesis with the lowest error.
        min_error = 1;        
        for j=1:M
            if(error(j) > 0.5)
                error(j) = 1 - error(j);
                ineq(j) = -1;
            else
                ineq(j) = +1;
            end

            if(error(j) < min_error)
                min_error = error(j);
                weak_learner.min_j = j;
                weak_learner.min_ineq = ineq(j);
                weak_learner.threshold = dec_stump(j);
            end
        end

        % Compute the alpha coefficient for each decision stump.
        weak_learner.alpha = 0.5 * log((1-min_error)/(min_error));

        %Update the distribution D.
        Z = 0;
        for i=1:N            
            D(i) = D(i)*exp(-weak_learner.alpha*labels(i)*hyp(i,weak_learner.min_j)*weak_learner.min_ineq);
            Z = Z + D(i);
        end
        
        if(strcmp(debug,'yes') == 1)
            D
        end
        
        D = D/Z;
        
        if(strcmp(debug,'yes') == 1)
            Z
            D
            weak_learner
        end

        %save the alpha, the value and dimension of the best decision stump, the sense of the inequality.         
        fileID = fopen(model_fname,'at');
        fmt = '%7.3f\t\t %d\t\t %d\t\t %7.3f\n';
        fprintf(fileID,fmt,[weak_learner.threshold weak_learner.min_ineq weak_learner.min_j weak_learner.alpha]);
        fclose(fileID);
    end
end