Data_num = 10; % 假定统计数据数量

% 假定输入数据矩阵(无实际数据，随机数取值)
s1 = zeros(1, Data_num); s1(:) = [4200 4150 4100 4050 4000 3950 3900 3850 3800 3750];
s2 = zeros(1, Data_num); s2(:) = [50 55 60 65 70 75 80 85 90 95];
d1 = zeros(1, Data_num); d1(:) = [60 61 61 62 62 63 63 64 64 65];
d2 = zeros(1, Data_num); d2(:) = [150 148 145 143 140 138 135 133 130 128];
d3 = zeros(1, Data_num); d3(:) = [0.8 0.82 0.85 0.9 0.95 1.0 1.02 1.03 1.01 1.00];
cwssi = zeros(Data_num,1); cwssi(:) = [0.35,0.245371102,0.198983311,0.12044137,0.091444884,0.012902942,-0.042180501,-0.155505051,-0.245371102,-0.376086957];
% 引入初始数据（略）
% 标准化
s1_dat = (s1 - min(s1)) / (max(s1) - min(s1));
s2_dat = (s2 - min(s2)) / (max(s2) - min(s2));
d1_dat = (max(d1) - d1) / (max(d1) - min(d1)); 
d2_dat = (max(d2) - d2) / (max(d2) - min(d2));
d3_dat = (max(d3) - d3) / (max(d3) - min(d3));

% 合并矩阵
data_total = [s1_dat; s2_dat; d1_dat; d2_dat; d3_dat]';

% 提取主成分
[coeff, score, latent, tsquared, explained, mu] = pca(data_total);

% 获取特征值（latent）
disp('特征值:');
disp(latent);

% 筛选特征值大于 1 的主成分（Kaiser准则）
kaiser_criteria = latent > 1;

% 获取符合Kaiser准则的主成分数量
num_kaiser = sum(kaiser_criteria);
disp(['符合特征值 > 1 的主成分数量: ', num2str(num_kaiser)]);

% 累计方差贡献率
cumulative_explained = cumsum(explained);
disp('累计方差贡献率:');
disp(cumulative_explained);

% 根据累计方差贡献率 ≥ 85% 选择的主成分数量
num_components_85 = find(cumulative_explained >= 85, 1);
disp(['根据累计方差贡献率≥85%，选择的主成分数量: ', num2str(num_components_85)]);

% 处理特殊情况：Kaiser准则和累计方差贡献率不一致时
if num_kaiser ~= num_components_85
    disp('警告: Kaiser准则和累计方差贡献率≥85%的主成分选择数量不一致。');
    disp(['符合Kaiser准则的主成分数量: ', num2str(num_kaiser)]);
    disp(['根据累计方差贡献率≥85%选择的主成分数量: ', num2str(num_components_85)]);
    disp('为确保选择的主成分符合两个标准，选择最小的主成分数量。');
end

% 确保选择的主成分数量不超过符合Kaiser准则的数量
num_components_final = min(num_components_85, num_kaiser);

% 筛选符合条件的主成分
kaiser_selected_coeff = coeff(:, kaiser_criteria);
selected_coeff = kaiser_selected_coeff(:, 1:num_components_final);

% 提取对应的主成分得分
selected_score = score(:, 1:num_components_final);

% 显示最终选择的主成分数量
disp(['最终选择的主成分数量: ', num2str(num_components_final)]);
% 供给和需求的指标载荷系数的归一化

% 假设我们对前两个主成分用于供给指标，后三个主成分用于需求指标

% 初始化供给和需求综合得分
S_total = zeros(Data_num, 1);
D_total = zeros(Data_num, 1);

% 遍历所有筛选的主成分
for i = 1:num_components_final
    % 供给指标的主成分载荷系数加权
    % 供给指标对应的载荷系数（s1 和 s2）与主成分得分相乘并加权求和
    S_total = S_total + coeff(1, i) * score(:, i);  % s1的权重为coeff(1,i)
    S_total = S_total + coeff(2, i) * score(:, i);  % s2的权重为coeff(2,i)
    
    % 需求指标的主成分载荷系数加权
    % 需求指标对应的载荷系数（d1, d2, d3）与主成分得分相乘并加权求和
    D_total = D_total + coeff(3, i) * score(:, i);  % d1的权重为coeff(3,i)
    D_total = D_total + coeff(4, i) * score(:, i);  % d2的权重为coeff(4,i)
    D_total = D_total + coeff(5, i) * score(:, i);  % d3的权重为coeff(5,i)
end

disp('供给综合得分 S_total:');
disp(S_total);
disp('需求综合得分 D_total:');
disp(D_total);
total_data=[S_total,D_total];

%构建线性回归模型CWSSI=α×Stotal+β×Dtotal+ε
cwssi=zeros(Data_num,1);
[B, BINT, R, RINT, STATS]=mvregress(total_data,cwssi);
%B1α，B2β，Re
disp('系数α，β为');
disp(B);
disp('ε为');
disp(R);
disp('R2');
disp(STATS(1));

%灰度模型构建GM（1，1）
[range_1,u1_1,u2_1,xt_1,yuce_1,epsilon_1,delta_1,rho_1]=GM(s1);
[range_2,u1_2,u2_2,xt_2,yuce_2,epsilon_2,delta_2,rho_2]=GM(s2);
[range_3,u1_3,u2_3,xt_3,yuce_3,epsilon_3,delta_3,rho_3]=GM(d1);
[range_4,u1_4,u2_4,xt_4,yuce_4,epsilon_4,delta_4,rho_4]=GM(d2);
[range_5,u1_5,u2_5,xt_5,yuce_5,epsilon_5,delta_5,rho_5]=GM(d3);

%模型耦合+优化

% 假定年份索引
years = (1:Data_num)';

% ------------- 1) 若已有观测CWSSI则使用，否则生成示例观测（便于演示） -------------
if exist('cwssi_obs','var') && length(cwssi_obs)==Data_num
    cwssi_obs = cwssi_obs(:);
    disp('使用输入的真实 cwssi_obs 作为观测序列。');
else
    % 如果没有真实观测，生成一个示例观测：用线性组合 + 噪声
    disp('未找到 cwssi_obs（观测值）。脚本将生成示例 cwssi_obs 用于演示。请用真实观测替换。');
    % 先用全局回归（若 B 已计算成功，则用 B，否则随机权重）
    try
        B_global = B; % 来自之前 mvregress 计算（可能失败）
    catch
        B_global = [0.5; -0.3;]; % 占位
    end
    % 生成观测 = B(1)*S_total + B(2)*D_total + 高斯噪声
    cwssi_obs = B_global(1)*S_total + B_global(2)*D_total + 0.05*randn(Data_num,1);
end

% ------------- 2) 全局回归（若之前mvregress可用则结果已在 B 中） -------------
% 为健壮性，使用增广设计矩阵含常数项做普通最小二乘回归（OLS）
X = [S_total, D_total, ones(Data_num,1)];
B_ols = X\cwssi_obs;  % [alpha; beta; intercept]
cwssi_pred_global = X * B_ols;

% 计算全局回归残差与R2
res_global = cwssi_obs - cwssi_pred_global;
SS_res = sum(res_global.^2);
SS_tot = sum((cwssi_obs - mean(cwssi_obs)).^2);
R2_global = 1 - SS_res/SS_tot;

disp('全局 OLS 回归系数 [alpha; beta; intercept]：');
disp(B_ols');
disp(['全局回归 R^2 = ', num2str(R2_global)]);

% ------------- 3) K-Means 聚类（优化回归系数） -------------
num_clusters = 3; % 可改为 3 或 4
opts = statset('UseParallel',false,'Display','off');
[cluster_idx, C] = kmeans([S_total, D_total], num_clusters, 'Replicates',5, 'Options',opts);

% 为每个簇拟合局部回归（含常数项）
B_cluster = zeros(3, num_clusters); % 每列为 [alpha; beta; intercept]
R2_cluster = zeros(num_clusters,1);
cwssi_pred_cluster = zeros(Data_num,1);

for k = 1:num_clusters
    idx_k = find(cluster_idx==k);
    if length(idx_k) < 3
        % 数据点太少时退回到全局系数
        B_cluster(:,k) = B_ols;
        R2_cluster(k) = NaN;
        cwssi_pred_cluster(idx_k) = cwssi_pred_global(idx_k);
        continue;
    end
    Xk = [S_total(idx_k), D_total(idx_k), ones(length(idx_k),1)];
    yk = cwssi_obs(idx_k);
    bk = Xk\yk;
    B_cluster(:,k) = bk;
    yk_pred = Xk*bk;
    resk = yk - yk_pred;
    R2_cluster(k) = 1 - sum(resk.^2)/sum((yk-mean(yk)).^2);
    % 将簇内预测填回总体预测序列
    cwssi_pred_cluster(idx_k) = [S_total(idx_k), D_total(idx_k), ones(length(idx_k),1)] * bk;
end

disp('每簇回归系数 [alpha beta intercept]：');
for k=1:num_clusters
    disp(['簇',num2str(k),': ', num2str(B_cluster(:,k)') '  R2=' num2str(R2_cluster(k))]);
end

% 动态 CWSSI（簇化回归预测）
cwssi_pred_bycluster = cwssi_pred_cluster;

% ------------- 4) 误差修正：用 GM 的 epsilon（各指标）合并作为自变量拟合回归残差 -------------
% 合并 GM 的 epsilon（注意 GM 返回的 epsilon 是行向量 x0'-yuce）
% 我们将 epsilon_1..5 合并取平均（或加权平均可替换）
% 保证 epsilon_* 长度与 Data_num 一致
epsilon_mat = [];
try
    epsilon_mat = [epsilon_1(:), epsilon_2(:), epsilon_3(:), epsilon_4(:), epsilon_5(:)];
catch
    % 若某些 epsilon 名称不在工作区，则创建零矩阵作为占位
    disp('未找到所有 GM 的 epsilon_* 变量，使用可用的 epsilon 或置零占位。');
    % 尝试收集已存在的 epsilon_*
    names = {'epsilon_1','epsilon_2','epsilon_3','epsilon_4','epsilon_5'};
    for i=1:length(names)
        if exist(names{i},'var')
        tmp = eval(names{i});      % 先取出变量
        epsilon_mat(:, end+1) = tmp(:); 
        end
    end
    if isempty(epsilon_mat)
        epsilon_mat = zeros(Data_num,5);
    else
        % 如果列数不足补零
        if size(epsilon_mat,2) < 5
            epsilon_mat(:,end+1:5) = 0;
        end
    end
end

% 取合并残差：这里取各指标 epsilon 的均值（也可以用最大、加权等）
epsilon_combined = mean(epsilon_mat,2);

% 计算回归残差（使用簇化回归预测作为基准）
res_clustered = cwssi_obs - cwssi_pred_bycluster;

% 用 epsilon_combined 去拟合 res_clustered 的二次多项式： res = p*epsilon^2 + q*epsilon + r
% 注意：polyfit 的自变量为 epsilon_combined，因变量为 res_clustered
p_coeffs = polyfit(epsilon_combined, res_clustered, 2); % [p,q,r]

% 使用拟合多项式对回归预测进行修正
f_epsilon = polyval(p_coeffs, epsilon_combined);
cwssi_corrected = cwssi_pred_bycluster + f_epsilon;

% ------------- 5) 评估修正效果 -------------
res_after = cwssi_obs - cwssi_corrected;
R2_after = 1 - sum(res_after.^2)/sum((cwssi_obs - mean(cwssi_obs)).^2);

disp('误差修正多项式系数 [p q r] = ');
disp(p_coeffs);
disp(['修正前（簇化回归） R^2 = ', num2str(1 - sum(res_clustered.^2)/SS_tot)]);
disp(['修正后 R^2 = ', num2str(R2_after)]);

% ------------- 6) 可视化：CWSSI 时间变化曲线（原始、回归预测、修正后） -------------
figure('Name','CWSSI 时间变化曲线','NumberTitle','off');
plot(years, cwssi_obs, '-o', 'LineWidth',1.5); hold on;
plot(years, cwssi_pred_global, '--', 'LineWidth',1.2);
plot(years, cwssi_pred_bycluster, '-.', 'LineWidth',1.2);
plot(years, cwssi_corrected, ':', 'LineWidth',1.8);
xlabel('时间（序号）');
ylabel('CWSSI 值');
legend('观测 CWSSI','全局回归预测','簇化回归预测','修正后 CWSSI','Location','best');
title('CWSSI 时间变化：观测 vs 预测 vs 修正');
grid on;

% ------------- 7) （可选）输出每年所用簇与对应系数、修正项 -------------
results_table = table(years, cluster_idx, S_total, D_total, cwssi_obs, cwssi_pred_bycluster, f_epsilon, cwssi_corrected, ...
    'VariableNames', {'Year','Cluster','S_total','D_total','CWSSI_obs','CWSSI_pred_cluster','Correction_f_eps','CWSSI_corrected'});
disp('前 10 行 结果示例：');
disp(results_table(1:min(10,end),:));

% 将簇的系数输出为更可读形式
cluster_coeffs_readable = array2table(B_cluster', 'VariableNames', {'alpha','beta','intercept'});
cluster_coeffs_readable.Cluster = (1:num_clusters)';
disp('簇回归系数汇总（可读表）：');
disp(cluster_coeffs_readable);

% 保存结果（如需要）
% save('CWSSI_results.mat','results_table','B_ols','B_cluster','p_coeffs','R2_global','R2_after');
