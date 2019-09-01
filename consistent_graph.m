function [con_graph, E, A] = consistent_graph(W, knn_idx, self_b, cross_b, tol, tol2)
% Learn a consistent graph from multiple graphs.
% Inputs:
%   W - weight matrix of a graph
%   knn_idx - common kNN index for all views
% Optional Inputs:
%   tol, tol2 - the tolerance that determines convergence of algorithm
% Outputs:
%   con_graph - weight matrix of the learned unified graph
%   E - a cell matrix containing the inconsistent part of all views
%   A - a cell matrix containing the consistent part of all views
% See "Youwei Liang, Dong Huang, and Chang-Dong Wang. Consistency Meets 
% Inconsistency: A Unified Graph Learning Framework for Multi-view 
% Clustering. 2019 IEEE International Conference on Data Mining(ICDM)."
% for a detailed description of the algorithm.
% Author: Youwei Liang
% 2019/08/31

if nargin < 5
    tol = 1e-8; tol2 = 1e-6;
end
v = length(W);
if nnz(W{1})/numel(W{1}) < 0.4  % if W contains a large proportion of zeros, use sparse mode
    for i=1:v
        W{i} = sparse(W{i});
    end
    sparse_mode = true;
else
    for i=1:v
        W{i} = full(W{i});
    end
    sparse_mode = false;
end
v = length(W);
b = cross_b*ones(v) - diag(cross_b*ones(1,v)) + diag(self_b*ones(1,v));
b_coef = b + eye(v);
n = size(W{1}, 1);
baW = cell(v,1);
special_baW = cell(v,1);
true_baW = cell(v,1);
A = cell(v,1);
B = cell(v,1);
E = cell(v,1);
up_knn_idx = triu(knn_idx);

obj_change = 10;
iter = 0;
maxiter = 40;
changes = zeros(maxiter,1);
obj = zeros(maxiter,1);

zz = 2.^(0:v-1);
ww = 1:2^v-2; % alpha can't be all zeros, so -2
logww = log2(ww);
yy = ww(abs(floor(logww)-logww)>eps);
alpha_zeros_ones = de2bi([0,zz,yy]);
n_eye_coef = -eye(v);

% initialize A{i}, alpha, con_graph
alpha = ones(v,1) / v;
con_graph = W{1};
if sparse_mode
    D = sparse(n, n);
else
    D = zeros(n,n);
end
for i=1:v
    D = max(D, W{i});
    A{i} = full(W{i});
%     temp = A{i}(knn_idx);
%     A{i}(knn_idx) = temp.*(1.5 - rand(m,1));
%     A{i} = (A{i}+A{i}')/2;
%     A{i}(knn_idx) = temp.*(2 - 1.9*rand(m,1));
end
for i=1:v
    if sparse_mode
        A{i} = sparse(A{i});
    end
    A{i} = min(A{i}, D);
end
% con_graph = con_graph/v + rand(m,1);

warning('off','MATLAB:nearlySingularMatrix')

while obj_change > tol && iter < maxiter
    iter = iter + 1;
    % fix A{i}, update con_graph and alpha
    obj1 = 0;
    for i=1:v
        E{i} = W{i} - A{i};
        temp = norm(alpha(i)*A{i}-con_graph, 'fro');
        obj1 = obj1 + temp*temp;
    end
    coef = zeros(v);
    coef2 = zeros(v);
    for i=1:v
        for j=i:v
            coef(i,j) = sum(sum(A{i}.*A{j}));
            coef(j,i) = coef(i,j);
            coef2(i,j) = sum(sum(E{i}.*E{j}));
            coef2(j,i) = coef2(i,j);
        end
    end
    coef2 = coef2 .* b;
    
    obj2 = sum(sum(coef2 .* (alpha * alpha')));
    last_obj = obj1+obj2;
    obj(iter) = last_obj;
    
    % compute coefficient for the linear equation
    H = 2*(diag(diag(coef)) - coef/v + coef2);
    one = ones(v, 1);

    old_alpha = alpha;
    old_con_graph = con_graph;
    
    for i=1:1
        mpl = alpha_zeros_ones(i,:);
        coef3 = H .* ~mpl + n_eye_coef .* mpl;
        X = [coef3, one; 1-mpl, 0];
        temp_b = [zeros(v,1); 1];
        if det(X) == 0 % abs(det(X)) <= eps
            fprintf('*************')
            solution = pinv(X)*temp_b;
        else
            solution = X \ temp_b;
        end
        alpha = EProjSimplex_new(solution(1:v));
    end
%     fprintf('best_obj:%.3f\n', best_obj)
    con_graph = alpha(1)*A{1};
    for j=2:v
        con_graph = con_graph + alpha(j)*A{j};
    end
    con_graph = con_graph/v;
    
    alpha_change = norm(alpha-old_alpha, 'fro');
    con_graph_change = norm(con_graph-old_con_graph, 'fro');

    % fix con_graph and alpha, update A{i}
    alp_coef = alpha * alpha';
    coef = alp_coef .* b_coef;
    if sparse_mode
        commom_baW = sparse(n, n);
    else
        commom_baW = zeros(n,n);
    end
    for i=1:v
        baW{i} = cross_b*alpha(i)*W{i};
        special_baW{i} = self_b*alpha(i)*W{i};
        commom_baW = commom_baW + baW{i};
    end

    for i=1:v
        true_baW{i} = commom_baW-baW{i}+special_baW{i};
        temp = full(alpha(i)*(con_graph + true_baW{i}));
        B{i} = temp(up_knn_idx);
    end
    right_b = cat(2, B{:})';
    if det(coef) == 0
        solution = (pinv(coef) * right_b)';
        fprintf('------------')
    else
        solution = (coef \ right_b)';
    end
    solution(solution<0) = 0;
    A_change = 0;
    for i=1:v
        temp = solution(:,i);
        oldA = A{i};
        A{i} = zeros(n, n);
        A{i}(up_knn_idx) = temp;
        A{i} = max(A{i}, A{i}');
        A{i} = min(W{i}, A{i});
        if sparse_mode
            A{i} = sparse(A{i});
        end
        A_change = A_change + norm(oldA-A{i}, 'fro');
    end
    
    change = A_change + alpha_change + con_graph_change;
    if iter > 1
        obj_change = min(abs(obj(iter)-obj(1:iter-1)))/abs(obj(1) - obj(iter));
    end
    changes(iter) = change;
    if rem(iter, 5) == 0
%         fprintf('*%2d:%.2f | ', iter, obj_change)
    end
    if iter > 20
        tol = tol2;
    elseif iter > 30
        tol = tol2*10;
    end
end

warning('on','MATLAB:nearlySingularMatrix')
% plot(1:iter, obj(1:iter))
end
