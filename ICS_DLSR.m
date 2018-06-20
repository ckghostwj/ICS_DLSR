function [Q,E,obj] = ICS_DLSR(X,Y,Train_Lab,opts)
% code is written by Jie Wen
% If any problems, please contact: wenjie@hrbeu.edu.cn
% Please cite the reference:
% Jie Wen, Yong Xu, Zuoyong Li, Zhongli Ma, Yuanrong Xu, 
% Inter-class sparsity based discriminative least square regression [J],
% Neural Networks, 2018, doi: 10.1016/j.neunet.2018.02.002.

miu = opts.miu;
rho = opts.rho;
max_miu = 1e8;
lambda1 = opts.lambda1;
lambda2 = opts.lambda2;
lambda3 = opts.lambda3;
nnClass   = opts.nnClass;
max_iter  = opts.maxIter;
tol2 = 1e-2;
Q  = rand(nnClass,size(X,1));
E  = zeros(nnClass,size(X,2));
F  = Q*X;
C1 = zeros(size(F));
for iter = 1:max_iter
    % ----------- 更新Q ----------- %
    G1 = Y+E;
    G2 = F+C1/miu;
    Q  = (G1+miu*G2)*X'*inv(X*X'*(1+miu)+lambda3*eye(size(X,1)));
    % ---------- 更新 F ------------ %
    H = Q*X-C1/miu;
    F = [];
    obj_F = 0;
    for ic = 1:nnClass
        idx = find(Train_Lab == ic);
        H_ic = H(:,idx);
        linshi_F = solve_l1l2(H_ic',lambda1/miu)';
        F = [F,linshi_F];
        obj_F = obj_F+sum(sqrt(sum(linshi_F.^2,2)));
    end
    clear linshi_F;
    % ---------- 更新 E ----------- %
    K = Q*X-Y;
    E = solve_l1l2(K',lambda2)';
    % ---------- C1 ------------ %
    Lq = F-Q*X;
    C1 = C1 + miu*Lq; 
    miu = min(rho*miu,max_miu);
   
    obj(iter) = 0.5*norm(Y+E-Q*X,'fro').^2+lambda3*0.5*norm(Q,'fro')^2+lambda1*obj_F+lambda2*sum(sqrt(sum(E.^2,2)));
    if iter > 3 && abs(obj(iter)-obj(iter-1)) < 1e-7
        iter
        break;
    end  
end
    
    