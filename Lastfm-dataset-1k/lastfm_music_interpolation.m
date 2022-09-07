clear all
cd('/tudelft.net/staff-bulk/ewi/insy/MMC/maosheng/trend_filtering_simplicial/cvx')
cvx_setup
addpath('/tudelft.net/staff-bulk/ewi/insy/MMC/maosheng/trend_filtering_simplicial/Lastfm-dataset-1k')
%% transition by song
t1 = readtable('B1.csv');
t2 = readtable('B2t.csv');
t3 = readtable('flow_vec.csv');
num_nodes = 3092; num_edges = 5507; num_tri = 571;

%% transition by artist
t1 = readtable('B1-artist.csv');
t2 = readtable('B2t-artist.csv');
t3 = readtable('flow_vec-artist.csv');
num_nodes = 657; num_edges = 1997; num_tri = 1276;

%% build in matlab
B1 = t1{:,:}; B2t = t2{:,:}; B2 = B2t';
L1l = B1'*B1; L1u = B2*B2t;
L1 = L1l + L1u;
I = eye(num_edges);
% the original data can be used for interpolation, where the filtering
% method does not perform well though
f = t3{:,:};
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
        f_filtered1 = I*f_in;
        corr_out_single(i,ii) = corr(f,f_filtered1);
        div_in(i,ii) = norm(B1*f_in);

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
        f_unlabeled = pinv([B1*expanding_matrix;0.5*eye(nnz(mask_un))])...
            *[-B1*f_in;zeros(nnz(mask_un),1)];
        f_jia2019(:,i,ii) = f_in + expanding_matrix*f_unlabeled;
        corr_jia2019(i,ii) = corr(f,f_jia2019(:,i,ii));
        div_jia2019(i,ii) = norm(B1*f_jia2019(:,i,ii));
        err_jia2019(i,ii) = norm(f-f_jia2019(:,i,ii))/norm(f);

        %% l1 norm based regularization
        % here we shall the constraint to preserve the observed labels
        cvx_begin
        variable f_opt_1(num_edges);
        minimize( 1*norm(f_opt_1-f_in) + 0.5* norm(B1*f_opt_1,1) );
        subject to
        sampling_matrix*f_opt_1 == sampling_matrix*f_in;
        cvx_end
        f_l1_1(:,i,ii) = f_opt_1;
        err_l1_1(i,ii) = norm(f_l1_1(:,i,ii)-f)/norm(f);
        corr_l1_1(i,ii) = corr(f,f_l1_1(:,i,ii));
        div_l1_1(i,ii) = norm(B1*f_l1_1(:,i,ii));

        cvx_begin
        variable f_opt_2(num_edges);
        minimize( 1*norm(f_opt_2-f_in) + 0.5*norm(B1*L1l*f_opt_2,1) );
        subject to
        sampling_matrix*f_opt_2 == sampling_matrix*f_in;
        cvx_end
        f_l1_2(:,i,ii) = f_opt_2;
        err_l1_2(i,ii) = norm(f_l1_2(:,i,ii)-f)/norm(f);
        corr_l1_2(i,ii) = corr(f,f_l1_2(:,i,ii));
        div_l1_2(i,ii) = norm(B1*f_l1_2(:,i,ii));

        cvx_begin
        variable f_opt_3(num_edges);
        minimize( 1*norm(f_opt_3-f_in) + 0.5* norm(B1*L1l^2*f_opt_3,1) );
        subject to
        sampling_matrix*f_opt_3 == sampling_matrix*f_in;
        cvx_end
        f_l1_3(:,i,ii) = f_opt_3;
        err_l1_3(i,ii) = norm(f_l1_3(:,i,ii)-f)/norm(f);
        corr_l1_3(i,ii) = corr(f,f_l1_3(:,i,ii));
        div_l1_3(i,ii) = norm(B1*f_l1_3(:,i,ii));
        
        cvx_begin
        variable f_opt_4(num_edges);
        minimize( 1*norm(f_opt_4-f_in) + 0.5* norm(L1l*f_opt_4,1) );
        subject to
        sampling_matrix*f_opt_4 == sampling_matrix*f_in;
        cvx_end
        f_l1_4(:,i,ii) = f_opt_4;
        err_l1_4(i,ii) = norm(f_l1_4(:,i,ii)-f)/norm(f);
        corr_l1_4(i,ii) = corr(f,f_l1_4(:,i,ii));
        div_l1_4(i,ii) = norm(B1*f_l1_4(:,i,ii));

        cvx_begin
        variable f_opt_5(num_edges);
        minimize( 1*norm(f_opt_5-f_in) + 0.5* norm(L1l^2*f_opt_5,1) );
        subject to
        sampling_matrix*f_opt_5 == sampling_matrix*f_in;
        cvx_end
        f_l1_5(:,i,ii) = f_opt_5;
        err_l1_5(i,ii) = norm(f_l1_5(:,i,ii)-f)/norm(f);
        corr_l1_5(i,ii) = corr(f,f_l1_5(:,i,ii));
        div_l1_5(i,ii) = norm(B1*f_l1_5(:,i,ii));
        
    end

end
corr_in = mean(corr_in_single,2);
corr_out = mean(corr_out_single,2);
corr_jia2019_mean = mean(corr_jia2019,2);
corr_l1_1_mean = mean(corr_l1_1,2);
corr_l1_2_mean = mean(corr_l1_2,2);
corr_l1_3_mean = mean(corr_l1_3,2);
corr_l1_4_mean = mean(corr_l1_4,2);
corr_l1_5_mean = mean(corr_l1_5,2);

err_l1_1_mean = mean(err_l1_1,2);
err_l1_2_mean = mean(err_l1_2,2);
err_l1_3_mean = mean(err_l1_3,2);
err_l1_4_mean = mean(err_l1_4,2);
err_l1_5_mean = mean(err_l1_5,2);
err_jia2019_mean = mean(err_jia2019,2);

div_l1_1_mean = mean(div_l1_1,2);
div_l1_2_mean = mean(div_l1_2,2);
div_l1_3_mean = mean(div_l1_3,2);
div_l1_4_mean = mean(div_l1_4,2);
div_l1_5_mean = mean(div_l1_5,2);
div_jia2019_mean = mean(div_jia2019,2);
div_in_mean = mean(div_in,2);
%%
filename = 'lastfm_interpolation.mat';
save(filename)

% %%
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
% plot(ratio, div_jia2019_mean,'--','LineWidth',2.5); hold on;
% plot(ratio, div_l1_mean,'LineWidth',2);
% legend( 'ssl', 'l1');
% set(gca, 'YScale', 'log')
% set(gca,'fontsize',14)
% xlabel('Ratio unlabeled')