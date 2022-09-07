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
%%
num_realizations = 100;
snr_db = -6:0.5:12;
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
        mu = 2;
        f_l2(:,i,ii) = (I+mu*L1u)\f_noisy;
        err_l2(i,ii) = norm(f_l2(:,i,ii) -f)/norm(f);
        div_l2(i,ii) = norm(B1*f_l2(:,i,ii));
        curl_l2(i,ii) = norm(B2'*f_l2(:,i,ii));
        %% consider l1 denoising
        cvx_begin
        variables f_opt_1(num_edges);
        minimize(1* norm(f_noisy-f_opt_1)+2*norm(B2'*f_opt_1,1));
        cvx_end
        f_l1_1(:,i,ii) = f_opt_1;
        err_l1_1(i,ii) = norm(f_l1_1(:,i,ii) -f)/norm(f);
        div_l1_1(i,ii) = norm(B1*f_l1_1(:,i,ii) );
        curl_l1_1(i,ii) = norm(B2'*f_l1_1(:,i,ii) );

        cvx_begin
        variables f_opt_2(num_edges);
        minimize(1* norm(f_noisy-f_opt_2)+2*norm(B2'*L1u*f_opt_2,1));
        cvx_end
        f_l1_2(:,i,ii) = f_opt_2;
        err_l1_2(i,ii) = norm(f_l1_2(:,i,ii) -f)/norm(f);
        div_l1_2(i,ii) = norm(B1*f_l1_2(:,i,ii) );
        curl_l1_2(i,ii) = norm(B2'*f_l1_2(:,i,ii) );

        cvx_begin
        variables f_opt_3(num_edges);
        minimize(1* norm(f_noisy-f_opt_3)+2*norm(B2'*L1u^2*f_opt_3,1));
        cvx_end
        f_l1_3(:,i,ii) = f_opt_3;
        err_l1_3(i,ii) = norm(f_l1_3(:,i,ii) -f)/norm(f);
        div_l1_3(i,ii) = norm(B1*f_l1_3(:,i,ii) );
        curl_l1_3(i,ii) = norm(B2'*f_l1_3(:,i,ii) );

        cvx_begin
        variables f_opt_4(num_edges);
        minimize(1* norm(f_noisy-f_opt_4)+2*norm(L1u*f_opt_4,1));
        cvx_end
        f_l1_4(:,i,ii) = f_opt_4;
        err_l1_4(i,ii) = norm(f_l1_4(:,i,ii) -f)/norm(f);
        div_l1_4(i,ii) = norm(B1*f_l1_4(:,i,ii) );
        curl_l1_4(i,ii) = norm(B2'*f_l1_4(:,i,ii) );

        cvx_begin
        variables f_opt_5(num_edges);
        minimize(1* norm(f_noisy-f_opt_5)+2*norm(L1u^2*f_opt_5,1));
        cvx_end
        f_l1_5(:,i,ii) = f_opt_5;
        err_l1_5(i,ii) = norm(f_l1_5(:,i,ii) -f)/norm(f);
        div_l1_5(i,ii) = norm(B1*f_l1_5(:,i,ii) );
        curl_l1_5(i,ii) = norm(B2'*f_l1_5(:,i,ii) );

    end
end
%%
curl_l1_1_mean = mean(curl_l1_1,2);
curl_l1_2_mean = mean(curl_l1_2,2);
curl_l1_3_mean = mean(curl_l1_3,2);
curl_l1_4_mean = mean(curl_l1_4,2);
curl_l1_5_mean = mean(curl_l1_5,2);
curl_l2_mean = mean(curl_l2,2);
curl_noisy_mean = mean(curl_noisy,2);

err_l1_1_mean = mean(err_l1_1,2);
err_l1_2_mean = mean(err_l1_2,2);
err_l1_3_mean = mean(err_l1_3,2);
err_l1_4_mean = mean(err_l1_4,2);
err_l1_5_mean = mean(err_l1_5,2);
err_l2_mean = mean(err_l2,2);
err_noisy_mean = mean(err_noisy,2);

filename = 'forex_denoise.mat';
save(filename)
%%
% figure;
% subplot(1,2,1);
% plot(snr_db,err_l2_mean,'--','LineWidth',3.5); hold on;
% plot(snr_db,err_l1_1_mean,'LineWidth',2); hold on;
% plot(snr_db,err_l1_2_mean,'LineWidth',2); hold on;
% plot(snr_db,err_l1_3_mean,'LineWidth',2); hold on;
% plot(snr_db,err_l1_4_mean,'LineWidth',2); hold on;
% plot(snr_db,err_l1_5_mean,'LineWidth',2); hold on;
% plot(snr_db,err_noisy_mean,'k','LineWidth',2)
% xlim([-6,12])
% legend('$\ell_2$ norm', '$\ell_1$ norm','noisy','Interpreter','latex');
% set(gca, 'YScale', 'log')
% set(gca,'fontsize',14)
% xlabel('SNR')
% 
% subplot(1,2,2);
% plot(snr_db,curl_l2_mean,'LineWidth',2); hold on;
% plot(snr_db, curl_l1_1_mean,'LineWidth',2)
% plot(snr_db, curl_l1_2_mean,'LineWidth',2)
% plot(snr_db, curl_l1_3_mean,'LineWidth',2)
% plot(snr_db, curl_l1_4_mean,'LineWidth',2)
% plot(snr_db, curl_l1_5_mean,'LineWidth',2)
% 
% xlim([-6,12])
% legend('$\ell_2$ norm', '$\ell_1$ norm','Interpreter','latex');
% set(gca, 'YScale', 'log')
% set(gca,'fontsize',14)
% xlabel('SNR')

