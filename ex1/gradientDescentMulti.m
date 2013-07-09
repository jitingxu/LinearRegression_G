function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
	
	%gradient with respect to theta 
	X_grad = [zeros(m,1) X(:,(2:3))];
	
	fprintf('First 20 examples from the dataset: \n');
	fprintf(' x = [%.0f %.0f %.0f], y = %.0f \n', [X_grad(1:10,:) y(1:10,:)]');

	fprintf('Program paused. Press enter to continue.\n');
	%pause;

	theta = theta - alpha * ((X * theta - y)' * X_grad)' / m;


	fprintf(' theta = [%.0f %.0f %.0f] \n', theta);

	fprintf('Program paused. Press enter to continue.\n');
	%pause;







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
