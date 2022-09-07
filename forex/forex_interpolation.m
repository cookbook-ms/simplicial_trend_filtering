clc;
clear;
close all;
cd('/tudelft.net/staff-bulk/ewi/insy/MMC/maosheng/trend_filtering_simplicial/cvx')
cvx_setup
addpath('/tudelft.net/staff-bulk/ewi/insy/MMC/maosheng/trend_filtering_simplicial/forex')
%rng(1223)
t1 = readtable('B1_FX_1538755200.csv');
t2 = readtable('B2t_FX_1538755200.csv');
t4 = readtable('flow_FX_1538755200.csv');
B1 = t1{:,:};
B2t = t2{:,:};
B2 = B2t';
% the edge flow is the mid of the ask and bid prices
f = t4{:,1};
% Hodge Laplacian
L1l = B1'*B1; L1u = B2*B2';
L1 = L1l + L1u;
% edge Laplacian
Le = L1l;
% graph Laplacian
L0 = B1*B1';
% triangle Laplacian
L2 = B2'*B2;
num_nodes = size(B1,1); num_edges = size(B1,2); num_tri = size(B2,2);
I = eye(num_edges);
[u1, lam]= eig(L1); eig_L1 = diag(lam);
% check the divergence and curl of the original signal
div_f = norm(B1*f); % the actual transition signal is div-free
curl_f = norm(B2'*f); % it is not curl-free
%% interpolation task
ratio = 0.05:0.05:1;
for i = 1:length(ratio)
    M = floor(num_edges*ratio(i)); % the number of nonzero nodes
    for ii = 1:100
        mask = zeros(num_edges,1);
        mask(randperm(numel(mask), M)) = 1;
        sampling_matrix = zeros(M,num_edges);
        sampling_matrix_1 = diag(mask);
        sampling_matrix = sampling_matrix_1(any(sampling_matrix_1,2),:);
        % the labeled
        f_in = f.*mask;
        corr_in_single(i,ii) = corr(f,f_in);
        % the filter method
        f_out = I*f_in;
        corr_out_single(i,ii) = corr(f,f_out);
        % curls
        curl_in(i,ii) = norm(B2'*f_in);
        curl_out(i,ii) = norm(B2'*f_out);

        %% the approach in Jia2019
        mask_un = not(mask);
        % build the expanding matrix
        expanding_matrix = zeros(num_edges,nnz(mask_un));
        row_ind = find(mask_un==1);
        for j = 1:length(row_ind)
            expanding_matrix(row_ind(j),j) = 1;
        end
        % pseudo inverse method, which leads to bad performance, slightly
        % better than zero-fill
        f_unlabeled = pinv([B2'*expanding_matrix;2*eye(nnz(mask_un))])...
            *[-B2'*f_in;zeros(nnz(mask_un),1)];
        % let us try the LSQR method
        f_jia2019(:,i,ii) = f_in + expanding_matrix*f_unlabeled;
        corr_jia2019(i,ii) = corr(f,f_jia2019(:,i,ii));
        count_jia2019(i,ii) =  nnz(abs(f-f_jia2019(:,i,ii)) < 0.1*f)/num_edges;
        curl_jia2019(i,ii) = norm(B2'*f_jia2019(:,i,ii));
        %% l1 norm based regularization
        % here we shall the constraint to preserve the observed labels
        cvx_begin
        variable f_opt_1(num_edges);
        minimize( 1*norm(f_opt_1-f_in) + 2* norm(B2'*f_opt_1,1) );
        subject to
        sampling_matrix*f_opt_1 == sampling_matrix*f_in;
        cvx_end
        f_l1_1(:,i,ii) = f_opt_1;
        corr_l1_1(i,ii) = corr(f,f_l1_1(:,i,ii));
        count_l1_1(i,ii) = nnz(abs(f-f_l1_1(:,i,ii)) < 0.1*f)/num_edges;
        curl_l1_1(i,ii) = norm(B2'*f_l1_1(:,i,ii));

        cvx_begin
        variable f_opt_2(num_edges);
        minimize( 1*norm(f_opt_2-f_in) + 2* norm(B2'*f_opt_2,1) );
        subject to
        sampling_matrix*f_opt_2 == sampling_matrix*f_in;
        cvx_end
        f_l1_2(:,i,ii) = f_opt_2;
        corr_l1_2(i,ii) = corr(f,f_l1_2(:,i,ii));
        count_l1_2(i,ii) = nnz(abs(f-f_l1_2(:,i,ii)) < 0.1*f)/num_edges;
        curl_l1_2(i,ii) = norm(B2'*f_l1_2(:,i,ii));

        cvx_begin
        variable f_opt_3(num_edges);
        minimize( 1*norm(f_opt_3-f_in) + 2* norm(B2'*f_opt_3,1) );
        subject to
        sampling_matrix*f_opt_3 == sampling_matrix*f_in;
        cvx_end
        f_l1_3(:,i,ii) = f_opt_3;
        corr_l1_3(i,ii) = corr(f,f_l1_3(:,i,ii));
        count_l1_3(i,ii) = nnz(abs(f-f_l1_3(:,i,ii)) < 0.1*f)/num_edges;
        curl_l1_3(i,ii) = norm(B2'*f_l1_3(:,i,ii));


        cvx_begin
        variable f_opt_4(num_edges);
        minimize( 1*norm(f_opt_4-f_in) + 2* norm(B2'*f_opt_4,1) );
        subject to
        sampling_matrix*f_opt_4 == sampling_matrix*f_in;
        cvx_end
        f_l1_4(:,i,ii) = f_opt_4;
        corr_l1_4(i,ii) = corr(f,f_l1_4(:,i,ii));
        count_l1_4(i,ii) = nnz(abs(f-f_l1_4(:,i,ii)) < 0.1*f)/num_edges;
        curl_l1_4(i,ii) = norm(B2'*f_l1_4(:,i,ii));

        cvx_begin
        variable f_opt_5(num_edges);
        minimize( 1*norm(f_opt_5-f_in) + 2* norm(B2'*f_opt_5,1) );
        subject to
        sampling_matrix*f_opt_5 == sampling_matrix*f_in;
        cvx_end
        f_l1_5(:,i,ii) = f_opt_5;
        corr_l1_5(i,ii) = corr(f,f_l1_5(:,i,ii));
        count_l1_5(i,ii) = nnz(abs(f-f_l1_5(:,i,ii)) < 0.1*f)/num_edges;
        curl_l1_5(i,ii) = norm(B2'*f_l1_5(:,i,ii));
    end
end
%%
curl_l1_1_mean = mean(curl_l1_1,2);
curl_l1_2_mean = mean(curl_l1_2,2);
curl_l1_3_mean = mean(curl_l1_3,2);
curl_l1_4_mean = mean(curl_l1_4,2);
curl_l1_5_mean = mean(curl_l1_5,2);
curl_jia2019_mean = mean(curl_jia2019,2);
curl_in_mean = mean(curl_in,2);
curl_out_mean = mean(curl_out,2);


count_l1_1_mean = mean(count_l1_1,2);
count_l1_2_mean = mean(count_l1_2,2);
count_l1_3_mean = mean(count_l1_3,2);
count_l1_4_mean = mean(count_l1_4,2);
count_l1_5_mean = mean(count_l1_5,2);
count_jia2019_mean = mean(count_jia2019,2);

corr_in= mean(corr_in_single,2);
corr_out = mean(corr_out_single,2);
corr_jia2019_mean = mean(corr_jia2019,2);
corr_l1_1_mean = mean(corr_l1_1,2);
corr_l1_2_mean = mean(corr_l1_2,2);
corr_l1_3_mean = mean(corr_l1_3,2);
corr_l1_4_mean = mean(corr_l1_4,2);
corr_l1_5_mean = mean(corr_l1_5,2);


filename = 'forex_interpolation.mat';
save(filename)
% figure;
% subplot(1,2,1);
% plot(ratio,corr_jia2019_mean,'--','LineWidth',3.5); hold on;
% plot(ratio,corr_l1_mean,'LineWidth',2); hold on;
% plot(ratio,corr_in,'k','LineWidth',2); hold on;
% legend( 'ssl', 'l1','zero fill')
% set(gca,'fontsize',14)
% xlabel('Ratio unlabeled')
%
% subplot(1,2,2);
% plot(ratio, curl_jia2019_mean,'--','LineWidth',2.5); hold on;
% plot(ratio, curl_l1_mean,'LineWidth',2);
% legend( 'ssl', 'l1');
% set(gca, 'YScale', 'log')
% set(gca,'fontsize',14)
% xlabel('Ratio unlabeled')