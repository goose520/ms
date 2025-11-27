Data_num = 100; % 假定统计数据数量

% 假定输入数据矩阵(无实际数据，随机数取值)
s1 = rand(1, Data_num); 
s2 = rand(1, Data_num);
d1 = rand(1, Data_num);
d2 = rand(1, Data_num);
d3 = rand(1, Data_num);
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

