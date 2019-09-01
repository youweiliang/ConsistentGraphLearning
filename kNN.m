function [A, idx] = kNN(B, knn)
[A, idx] = maxk(B, knn, 2, 'sorting', false);
n = size(A, 1);

% adjacency_matrix = zeros(n,n);
% for i=1:n
%     adjacency_matrix(i, idx(i,:)) = A(i, :);
% end
% A = sparse(adjacency_matrix);

rowidx = ones(knn, n) .* [1:n];
A = sparse(rowidx, idx', A', n, n);

% A = max(A, A');
A = (A + A')/2;
% A = power(A .* A', 1/2);
end