function [boost] = adaboost_train(Y, Yw, T)
% Runs AdaBoost for T rounds given the predictions of a set of weak learners.
%
% Usage:
%
%   boost = adaboost_train(Y, YW, T)
%
% Returns a struct containing the results of running AdaBoost for T rounds.
% The input Y should be a (-1,1) binary N x 1 vector of labels. The input
% YW should be a (-1,1) binary N x M matrix containing the predictions of M
% weak learners on the dataset. T is the number of rounds of boosting. The
% returned struct has the following fields:
%
%   boost.err - 1 x T vector of weighted error at round t
%   boost.train_err - 1 x T vector of cumulative training error
%   boost.h - 1 X T vector indicating which weak learner was chosen
%   boost.alpha - 1 X T vector of weights for combining learners
%
% NOTE: This implementation of Adaboost you are creating only works when
% you can precompute a pool of possible weak learners. In general, you
% might want to train a weak learner for each D rather than picking a
% precomputed one with minimal error.

% HINT: READ ALL THE HINTS AND DOCUMENTS WE GIVE YOU BEFORE BEGINNING.

% HINT: Look at ADABOOST_TEST before trying to implement AdaBoost so you
% can see how we expect this to be used.

% HINT: Precompute where each weak learner makes mistakes BEFORE running
% the main boosting loop. Then you can compute weighted error by weighting
% the mistakes of each learner.

% HINT: If predictions and labels are +1, -1, then errors are when the true 
% label multiplied by prediction is -1. 

% HINT: Follow the AdaBoost algorithm given in class, NOT THE ADABOOST
% ALGORITHM GIVEN IN BISHOP. DO NOT USE BISHOP FOR THIS.

% Perform any initialization or precomputation here.

% Initialize distribution over examples.
D = ones(size(Y));
D = 1./sum(D);
[N, M] = size(Yw);
Dist = zeros(T, length(Y)); %let distribution be a T * size(Y) matrix
Dist(1,:) = D;   %initialize D_1(j)

t0 = CTimeleft(T);
for t = 1:T
    t0.timeleft();
    %calculate the error rate, and select the min err, the corresponding
    %index is h(t)
    for i = 1 : M 
        temp(i) = Dist(t,i) * sum(Yw(:,i) ~= Y);          
    end
    [err(t), h(t)] = min(temp);
    %update alpha(t)
    alpha(t) = 1/2 * log((1-err(t))/err(t));
    
    %calculate Z(t);
    for k = 1 : M
        temp_z(t, k) = Dist(t,k) * exp(-alpha(t) .* Y(k) .* Yw(k, h(t)) );
    end
    Z(t) = sum(temp_z(t,:));
    train_err = sum(Y ~= Yw(:,h(t))) / N;
    %next distribution
    for j = 1 : M
        if(t < T)
            Dist(t+1,j) = Dist(t,j) * exp(-alpha(t).*Y(j) .* Yw(j, h(t))) /Z(t);
        end
    end
    
    % Compute the BEST weak learner according to current D, etc. -- put
    % AdaBoost logic here.  Make sure to update h(t), err(t), alpha(t), and
    % train_err(t). Note that h(t) should be the INDEX of the best weak
    % learner, and thus a single scalar number.

    % HINT: Make sure to normalize D so it sums to one!!
end

% Store results of boosting algorithm.
boost.train_err = train_err;
boost.err = err;
boost.h = h;
boost.alpha = alpha;