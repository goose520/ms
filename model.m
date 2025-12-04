clc,clear;
% 假定数据数量
Data_num = 12;

% 实际输入数据矩阵
s1 = zeros(1, Data_num); 
s1(:) = [2041.499, 2027.779, 2013.929, 2001.064, 1989.434, 1978.308, 1966.804, 1957.695, 1951.570, 1948.101, 1947.791, 1949.670];

s2 = zeros(1, Data_num); 
s2(:) = [7.17, 7.02, 8.30, 9.14, 9.30, 8.16, 8.15, 9.27, 8.19, 10.70, 10.58, 10.58];

d1 = zeros(1, Data_num); 
d1(:) = [56.784, 52.047, 53.188, 54.039, 55.616, 67.160, 69.248, 73.118, 77.047, 80.257, 80.244, 80.322];

d2 = zeros(1, Data_num); 
d2(:) = [18660, 18056, 17742, 16786, 16425, 16140, 15777, 15404, 14825, 11418, 9047, 8996];

d3 = zeros(1, Data_num); 
d3(:) = [1391242.343, 1400655.332, 1410287.993, 1419354.592, 1427652.016, 1435681.095, 1444079.121, 1450798.227, 1455351.282, 1457942.854, 1458174.603, 1456769.951];

cwssi = zeros(Data_num,1); 
cwssi(:) = [0.146, 0.134, 0.141, 0.167, 0.176, -0.046, -0.064, -0.129, -0.236, 0.036, 0.040, 0.037];

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


% 累计方差贡献率
cumulative_explained = cumsum(explained);
disp('累计方差贡献率:');
disp(cumulative_explained);

% 根据累计方差贡献率 ≥ 85% 选择的主成分数量
num_components_85 = find(cumulative_explained >= 85, 1);
disp(['根据累计方差贡献率≥85%，选择的主成分数量: ', num2str(num_components_85)]);


num_components_final = num_components_85;

% 筛选符合条件的主成分（仅用累计贡献率≥85%）
selected_coeff = coeff(:, 1:num_components_final);

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
if exist('cwssi','var') && length(cwssi)==Data_num
    cwssi = cwssi(:);
    disp('使用输入的真实 cwssi作为观测序列。');
else
    % 如果没有真实观测，生成一个示例观测：用线性组合 + 噪声
    disp('未找到 cwssi（观测值）。脚本将生成示例 cwssi 用于演示。请用真实观测替换。');
    % 先用全局回归（若 B 已计算成功，则用 B，否则随机权重）
    try
        B_global = B; % 来自之前 mvregress 计算（可能失败）
    catch
        B_global = [0.5; -0.3;]; % 占位
    end
    % 生成观测 = B(1)*S_total + B(2)*D_total + 高斯噪声
    cwssi = B_global(1)*S_total + B_global(2)*D_total + 0.05*randn(Data_num,1);
end

% ------------- 2) 全局回归（若之前mvregress可用则结果已在 B 中） -------------
% 为健壮性，使用增广设计矩阵含常数项做普通最小二乘回归（OLS）
X = [S_total, D_total, ones(Data_num,1)];
B_ols = X\cwssi;  % [alpha; beta; intercept]
cwssi_pred_global = X * B_ols;

% 计算全局回归残差与R2
res_global = cwssi - cwssi_pred_global;
SS_res = sum(res_global.^2);
SS_tot = sum((cwssi - mean(cwssi)).^2);
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
    yk = cwssi(idx_k);
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
res_clustered = cwssi - cwssi_pred_bycluster;

% 用 epsilon_combined 去拟合 res_clustered 的二次多项式： res = p*epsilon^2 + q*epsilon + r
% 注意：polyfit 的自变量为 epsilon_combined，因变量为 res_clustered
p_coeffs = polyfit(epsilon_combined, res_clustered, 2); % [p,q,r]

% 使用拟合多项式对回归预测进行修正
f_epsilon = polyval(p_coeffs, epsilon_combined);
cwssi_corrected = cwssi_pred_bycluster + f_epsilon;

% ------------- 5) 评估修正效果 -------------
res_after = cwssi - cwssi_corrected;
R2_after = 1 - sum(res_after.^2)/sum((cwssi - mean(cwssi)).^2);

disp('误差修正多项式系数 [p q r] = ');
disp(p_coeffs);
disp(['修正前（簇化回归） R^2 = ', num2str(1 - sum(res_clustered.^2)/SS_tot)]);
disp(['修正后 R^2 = ', num2str(R2_after)]);

% ------------- 6) 可视化：CWSSI 时间变化曲线（原始、回归预测、修正后） -------------
figure('Name','CWSSI 时间变化曲线','NumberTitle','off');
plot(years, cwssi, '-o', 'LineWidth',1.5); hold on;
plot(years, cwssi_pred_global, '--', 'LineWidth',1.2);
plot(years, cwssi_pred_bycluster, '-.', 'LineWidth',1.2);
plot(years, cwssi_corrected, ':', 'LineWidth',1.8);
xlabel('时间（序号）');
ylabel('CWSSI 值');
legend('观测 CWSSI','全局回归预测','簇化回归预测','修正后 CWSSI','Location','best');
title('CWSSI 时间变化：观测 vs 预测 vs 修正');
grid on;

% ------------- 7) （可选）输出每年所用簇与对应系数、修正项 -------------
results_table = table(years, cluster_idx, S_total, D_total, cwssi, cwssi_pred_bycluster, f_epsilon, cwssi_corrected, ...
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

%% ========= 生成未来20年（原始量纲）预测 S1_pre ~ D3_pre =========

future_years = 20;

% 建立未来时间 t
n_old = length(s1);
t_future = (n_old : n_old + future_years - 1)';

% GM(1,1) 的预测公式：xt(t) = (s(1) - b/a)*exp(-a*(t-1)) + b/a
GM_predict = @(a,b,first,x) (first - b/a) * exp(-a*(x-1)) + b/a;

% ------------ S1 ----------
a = double(u1_1); 
b = double(u2_1); 
S1_pre = GM_predict(a,b,s1(1), t_future);

% ------------ S2 ----------
a = double(u1_2); 
b = double(u2_2); 
S2_pre = GM_predict(a,b,s2(1), t_future);

% ------------ D1 ----------
a = double(u1_3); 
b = double(u2_3); 
D1_pre = GM_predict(a,b,d1(1), t_future);

% ------------ D2 ----------
a = double(u1_4); 
b = double(u2_4); 
D2_pre = GM_predict(a,b,d2(1), t_future);

% ------------ D3 ----------
a = double(u1_5); 
b = double(u2_5); 
D3_pre = GM_predict(a,b,d3(1), t_future);

fprintf("未来 20 年预测完成：S1_pre, S2_pre, D1_pre, D2_pre, D3_pre 已生成。\n");

%% ================== 未来20年 CWSSI_pre 与 WS_pre 计算 ===================

% 合并未来预测数据（原始量纲）
future_data = [S1_pre S2_pre D1_pre D2_pre D3_pre];

%% ======== 用历史 min/max 定义 amin/amax，并把 future_data 标准化（修复 amin 未定义错误） ========
% 假定 future_data 已经是 (future_len x 5) 格式： [S1_pre S2_pre D1_pre D2_pre D3_pre]
% 如果你尚未生成 S1_pre..D3_pre，请先运行 GM 预测段生成它们。

% --- 1) 计算历史 min/max （与脚本前面标准化一致）
amin = [ min(s1), min(s2), min(d1), min(d2), min(d3) ];  % 1x5
amax = [ max(s1), max(s2), max(d1), max(d2), max(d3) ];  % 1x5

% 检查是否有相等的 max 和 min（避免除以零）
zero_range = (amax - amin) == 0;
if any(zero_range)
    warning('Some variable(s) have zero range (max==min). Those columns will be left as zeros after normalization.');
    amax(zero_range) = amin(zero_range) + eps; % 避免除零
end

% --- 2) 标准化 future_data（按列 min-max，与历史口径一致）
% future_data 应为 (future_len x 5)
s_data_f = (future_data - amin) ./ (amax - amin);   % MATLAB 自动广播： (N x 5) - (1 x 5)

% --- 3) 将 future 标准化数据中心化并投影到 PCA（使用历史的 mu 和 coeff）
% 注意：pca 返回的 score = (data_centered) * coeff，且 mu 是历史样本均值（1x5）
if exist('mu','var') && exist('coeff','var')
    % mu is 1x5 (mean of historical standardized variables used in pca)
    % BUT 注意：你前面把 PCA 用在 data_total（已经是标准化过的），并把 mu 返回
    % 所以这里应先用与 PCA 相同的标准化口径（s_data_f），再减去 mu，然后乘 coeff。
    data_centered_f = s_data_f - mu;  % (future_len x 5) - (1 x 5)
    score_f = data_centered_f * coeff; % (future_len x 5) * (5 x 5) -> (future_len x 5)
else
    error('缺少 PCA 输出变量 coeff 或 mu：请先运行 PCA（[coeff,score,latent,tsquared,explained,mu]=pca(data_total)）');
end

% --- 4) 按照你历史脚本里相同的“供给/需求合成”方法计算 S_total_f, D_total_f
% （为保持一致性，这里沿用你原来的逐主成分加权求和方式）
future_len = size(score_f,1);
S_total_f = zeros(future_len,1);
D_total_f = zeros(future_len,1);

for i = 1:num_components_final
    % 注意：coeff(j,i) 是第 j 个原始变量在第 i 个主成分上的载荷
    % score_f(:,i) 是未来样本在主成分 i 上的得分（列向量）
    S_total_f = S_total_f + coeff(1,i) * score_f(:,i); % s1
    S_total_f = S_total_f + coeff(2,i) * score_f(:,i); % s2

    D_total_f = D_total_f + coeff(3,i) * score_f(:,i); % d1
    D_total_f = D_total_f + coeff(4,i) * score_f(:,i); % d2
    D_total_f = D_total_f + coeff(5,i) * score_f(:,i); % d3
end

% --- 5) 用历史训练得到的回归系数 B_ols 预测 CWSSI_pre（确保 B_ols 已存在）
if exist('B_ols','var')
    X_fut = [S_total_f, D_total_f, ones(future_len,1)]; % (future_len x 3)
    CWSSI_pre = X_fut * B_ols;  % (future_len x 1)
else
    warning('未找到 B_ols（OLS 回归系数）。将尝试使用之前 mvregress 的 B（如果有）或给出 NaN。');
    if exist('B','var') && numel(B)>=2
        Btemp = B;
        if length(Btemp)==2 % mvregress 有时返回 2系数 (无截距)
            CWSSI_pre = Btemp(1)*S_total_f + Btemp(2)*D_total_f;
        else
            CWSSI_pre = nan(future_len,1);
        end
    else
        CWSSI_pre = nan(future_len,1);
    end
end

% 如需将 CWSSI 约束到 [0,1]
CWSSI_pre = max(min(CWSSI_pre,1),0);

% --- 6) 计算 WS_pre（供需比），这里定义为 D_total_f ./ S_total_f *100（%）
% 如果你有“总取水量”（TotalWithdrawal），可以改为 TotalWithdrawal ./ S1_pre *100
WS_pre = nan(future_len,1);
zero_mask = (S_total_f ~= 0);
WS_pre(zero_mask) = (D_total_f(zero_mask) ./ S_total_f(zero_mask)) * 100;
WS_pre(~zero_mask) = NaN;  % 无法计算的放 NaN

% --- 7) 将生成的未来时间向量与结果绑定（可选）
future_years_vec = (max(years)+1 : max(years)+future_len)';
% 输出提示
fprintf('已生成 %d 年预测：CWSSI_pre (len=%d) 与 WS_pre (len=%d)。\n', future_len, length(CWSSI_pre), length(WS_pre));


% ② 做 PCA —— 必须使用历史样本的 coeff，而不能重新训练！
score_f = s_data_f * coeff;

% ③ 取前两个主成分
X1_fut = score_f(:,1);
Y1_fut = score_f(:,2);


%% ====== 若 sy/ux/uw/a2/b2/ intercept 未定义则自动用历史数据最小二乘估计 ======
% 需要历史变量： score (历史主成分得分), S_total, D_total, B_ols (或 B)
if ~exist('score','var')
    error('缺少历史主成分得分变量 score（PCA 输出）。请先运行 PCA 部分。');
end

% 历史主成分第一列/第二列
X1_hist = score(:,1);
Y1_hist = score(:,2);

% 估计 sy：最小二乘拟合 S_total ≈ X1_hist - sy * Y1_hist
if ~exist('sy','var')
    if exist('S_total','var')
        denom = (Y1_hist'*Y1_hist);
        if denom == 0
            warning('Y1_hist 全为0，不能估计 sy；将 sy 置为 0。');
            sy = 0;
        else
            sy = - (Y1_hist' * (S_total - X1_hist)) / denom;
        end
        fprintf('已估计 sy = %.6g\n', sy);
    else
        warning('无法估计 sy：缺少历史 S_total。将 sy 置为 0。');
        sy = 0;
    end
end

% 估计 ux, uw：最小二乘拟合 -D_total ≈ [X1_hist Y1_hist] * [ux; uw]
if ~(exist('ux','var') && exist('uw','var'))
    if exist('D_total','var')
        A = [X1_hist, Y1_hist];           % n x 2
        b = -D_total;                     % n x 1
        % 如果 A 的列近似线性相关，\ 仍给最小二乘解
        params = A \ b;                   % 2x1
        if length(params) == 2
            ux = params(1);
            uw = params(2);
            fprintf('已估计 ux = %.6g, uw = %.6g\n', ux, uw);
        else
            warning('估计 ux,uw 失败，置为零。');
            ux = 0; uw = 0;
        end
    else
        warning('无法估计 ux/uw：缺少历史 D_total。将 ux,uw 置为 0。');
        ux = 0; uw = 0;
    end
end

% 估计 a2,b2,intercept：优先使用已存在的 B_ols（[alpha; beta; intercept]）
if ~exist('a2','var') || ~exist('b2','var') || ~exist('intercept','var')
    if exist('B_ols','var') && numel(B_ols)>=3
        a2 = B_ols(1);
        b2 = B_ols(2);
        intercept = B_ols(3);
        fprintf('已采用 B_ols 中的回归系数 a2=%.6g, b2=%.6g, intercept=%.6g\n', a2, b2, intercept);
    elseif exist('B','var') && numel(B)>=2
        % B 可能是不含截距的 mvregress 输出
        if numel(B) == 2
            a2 = B(1); b2 = B(2); intercept = 0;
            fprintf('采用 mvregress 输出 B（无截距）: a2=%.6g, b2=%.6g\n', a2, b2);
        else
            a2 = NaN; b2 = NaN; intercept = NaN;
            warning('无法找到合适回归系数 B_ols 或 B，CWSSI_pre 将为 NaN。');
        end
    else
        % 最后一招：用历史数据作简单线性回归 CWSSI ~ S_total + D_total
        if exist('cwssi','var') && exist('S_total','var') && exist('D_total','var')
            Xhist = [S_total, D_total, ones(length(S_total),1)];
            beta_hist = Xhist \ cwssi;
            a2 = beta_hist(1); b2 = beta_hist(2); intercept = beta_hist(3);
            fprintf('用历史数据拟合得到 a2=%.6g, b2=%.6g, intercept=%.6g\n', a2, b2, intercept);
        else
            a2 = NaN; b2 = NaN; intercept = NaN;
            warning('无法估计回归系数：缺少 cwssi/S_total/D_total。CWSSI_pre 会是 NaN。');
        end
    end
end

% 检查是否都已定义为数值
if any(isnan([sy, ux, uw, a2, b2, intercept]))
    warning('估计出的某些系数为 NaN，后续 CWSSI_pre/WS_pre 可能包含 NaN 值。');
end



% ④ 求未来 S_total, D_total
S_total_f = X1_fut - sy * Y1_fut;
D_total_f = -(ux * X1_fut + uw * Y1_fut);

% ⑤ 用历史回归模型求未来 CWSSI
CWSSI_pre = a2 .* S_total_f + b2 .* D_total_f + 0.5;
CWSSI_pre = max(min(CWSSI_pre,1),0); % 限制 0~1

% 供需比 WS = S_total / D_total （或 S_total - D_total，看你之前定义）
WS_pre = S_total_f ./ D_total_f;

fprintf("未来 20 年 CWSSI_pre 与 WS_pre 已生成。\n");
plot_CWSSI_figures_full;