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
B1 = t1{:,:}; B2t = t2{:,:}; B2 = B2t'; B1*B2;
L1l = B1'*B1; L1u = B2*B2t;
L1 = L1l + L1u;
I = eye(num_edges);
% the original data can be used for interpolation, where the filtering
% method does not perform well though
f = t3{:,:};
% check the divergence and curl of the original signal
div_f = norm(B1*f); % the actual transition signal is div-free
curl_f = norm(B2'*f); % it is not curl-free

%%
num_realizations = 100;
snr_db = -12:0.5:6;
snr = 10.^(snr_db/10);
power_flow = norm(f,2);
power_noise = power_flow./snr/num_edges;
for i = 1:length(snr)
    for ii = 1:num_realizations
        noise = power_noise(i)*randn(num_edges,1);
        f_noisy = f + noise;
        div_noisy(i,ii) = norm(B1*f_noisy);
        curl_noisy(i,ii) = norm(B2'*f_noisy);
        err_noisy(i,ii) = norm(f_noisy-f)/norm(f);
        %% perform l2 denoising
        mu = 0.5;
        f_l2(:,i,ii) = (I+mu*L1l)\f_noisy;
        err_l2(i,ii) = norm(f_l2(:,i,ii)-f)/norm(f);
        div_l2(i,ii) = norm(B1*f_l2(:,i,ii));
        curl_l2(i,ii) = norm(B2'*f_l2(:,i,ii));
        %% consider l1 denoising

        cvx_begin
        variables f_opt_1(num_edges);
        minimize(1* norm(f_noisy-f_opt_1)+0.5*norm(B1*f_opt_1,1) );
        cvx_end
        f_l1_1(:,i,ii) = f_opt_1;
        err_l1_1(i,ii) = norm(f_opt_1-f)/norm(f);
        div_l1_1(i,ii) = norm(B1*f_l1_1(:,i,ii));
        curl_l1_1(i,ii) = norm(B2'*f_l1_1(:,i,ii));

         cvx_begin
        variables f_opt_2(num_edges);
        minimize(1* norm(f_noisy-f_opt_2)+0.5*norm(B1*L1l*f_opt_2,1) );
        cvx_end
        f_l1_2 = f_opt_2;
        err_l1_2(i,ii) = norm(f_l1_2-f)/norm(f);
        div_l1_2(i,ii) = norm(B1*f_l1_2);
        curl_l1_2(i,ii) = norm(B2'*f_l1_2);

         cvx_begin
        variables f_opt_3(num_edges);
        minimize(1* norm(f_noisy-f_opt_3)+0.5*norm(B1*L1l^2*f_opt_3,1) );
        cvx_end
        f_l1_3(:,i,ii) = f_opt_3;
        err_l1_3(i,ii) = norm(f_l1_3(:,i,ii)-f)/norm(f);
        div_l1_3(i,ii) = norm(B1*f_l1_3(:,i,ii));
        curl_l1_3(i,ii) = norm(B2'*f_l1_3(:,i,ii));

        cvx_begin
        variables f_opt_4(num_edges);
        minimize(1* norm(f_noisy-f_opt_4)+0.5*norm(L1l*f_opt_4,1) );
        cvx_end
        f_l1_4(:,i,ii) = f_opt_4;
        err_l1_4(i,ii) = norm(f_l1_4(:,i,ii) -f)/norm(f);
        div_l1_4(i,ii) = norm(B1*f_l1_4(:,i,ii));
        curl_l1_4(i,ii) = norm(B2'*f_l1_4(:,i,ii));

        cvx_begin
        variables f_opt_5(num_edges);
        minimize(1* norm(f_noisy-f_opt_5)+0.5*norm(L1l^2*f_opt_5,1) );
        cvx_end
        f_l1_5(:,i,ii) = f_opt_5;
        err_l1_5(i,ii) = norm(f_l1_5(:,i,ii) -f)/norm(f);
        div_l1_5(i,ii) = norm(B1*f_l1_5(:,i,ii));
        curl_l1_5(i,ii) = norm(B2'*f_l1_5(:,i,ii));
    end
end
%%
div_l1_1_mean = mean(div_l1_1,2);
div_l1_2_mean = mean(div_l1_2,2);
div_l1_3_mean = mean(div_l1_3,2);
div_l1_4_mean = mean(div_l1_4,2);
div_l1_5_mean = mean(div_l1_5,2);
div_l2_mean = mean(div_l2,2);
div_noisy_mean = mean(div_noisy,2);

err_l1_1_mean = mean(err_l1_1,2);
err_l1_2_mean = mean(err_l1_2,2);
err_l1_3_mean = mean(err_l1_3,2);
err_l1_4_mean = mean(err_l1_4,2);
err_l1_5_mean = mean(err_l1_5,2);
err_l2_mean = mean(err_l2,2);
err_noisy_mean = mean(err_noisy,2);

%%
filename = 'lastfm_denoise.mat';
save(filename)

%%
% figure;
% subplot(1,2,1);
% plot(snr_db,err_l2_mean,'--','LineWidth',3.5); hold on;
% plot(snr_db,err_l1_1_mean,err_l1_2_mean,err_l1_3_mean, ...
%     err_l1_4_mean,err_l1_5_mean,'LineWidth',2); hold on;
% plot(snr_db,err_noisy_mean,'k','LineWidth',2)
% xlim([-12,6])
% %legend('$\ell_2$ norm', '$\ell_1$ norm','noisy','Interpreter','latex');
% % set(gca, 'YScale', 'log')
% set(gca,'fontsize',14)
% xlabel('SNR')
% 
% subplot(1,2,2);
% plot(snr_db,div_l2_mean,'LineWidth',2); hold on;
% plot(snr_db, div_l1_1_mean, div_l1_2_mean, div_l1_3_mean,...
%     div_l1_4_mean,div_l1_5_mean,'LineWidth',2)
% xlim([-12,6])
% %legend('$\ell_2$ norm', '$\ell_1$ norm','Interpreter','latex');
% %set(gca, 'YScale', 'log')
% set(gca,'fontsize',14)
% xlabel('SNR')

