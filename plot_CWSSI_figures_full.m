function plot_CWSSI_figures_full()
% plot_CWSSI_figures_full
% 完整绘图脚本（无 Image Processing Toolbox 依赖）
% 产生：
%   1) CWSSI 模型框架图
%   2) 中国水资源短缺驱动因素框架图
%   3) 2041 年水情预测框架图
%   4) 2009-2041 年 CWSSI 及 5 个核心指标趋势图
%   5) 干预方案效果对比图
%
% 要求：将该文件与 model.m 放同一目录，在运行 model.m 后（或在 workspace
% 有必要变量时）直接运行 plot_CWSSI_figures_full。
%
% 该脚本对缺失变量会以 NaN 填充并给出警告，不会直接中断。

%% ------------- User settings -------------
% 如果历史年份不是默认的 2011-2022，可修改下面起始年份与长度
hist_start_year = 2011;   % 若你是 2009-2020，请改为 2009
hist_len = 12;            % 历史长度（默认12）
hist_years = (hist_start_year : hist_start_year + hist_len - 1)';

% 未来默认预测年数（若 workspace 中已有 CWSSI_pre/S1_pre 等将使用它们的长度）
future_len_default = 20;

% style
fontname = 'Times New Roman';
title_fontsize = 14;
label_fontsize = 11;
anno_fontsize = 10;
save_dpi = 300;

% Determine future_len from available workspace variables
if evalin('base','exist(''CWSSI_pre'',''var'')')
    tmp = evalin('base','CWSSI_pre(:)');
    future_len = length(tmp);
elseif evalin('base','exist(''S1_pre'',''var'')')
    tmp = evalin('base','S1_pre(:)');
    future_len = length(tmp);
else
    future_len = future_len_default;
end
pred_years = (hist_years(end)+1 : hist_years(end) + future_len)';

%% ------------- safe read helper -------------
% read variable from base workspace, else return NaNs of specified length
getVarOrNaN = @(varname, len) safe_get(varname, len);

% load historical series (try multiple common names)
S1_old = getVarOrNaN('S1_old', hist_len);
if all(isnan(S1_old)) && evalin('base','exist(''s1'',''var'')')
    S1_old = evalin('base','s1(:)');
end
S2_old = getVarOrNaN('S2_old', hist_len);
if all(isnan(S2_old)) && evalin('base','exist(''s2'',''var'')')
    S2_old = evalin('base','s2(:)');
end
D1_old = getVarOrNaN('D1_old', hist_len);
if all(isnan(D1_old)) && evalin('base','exist(''d1'',''var'')')
    D1_old = evalin('base','d1(:)');
end
D2_old = getVarOrNaN('D2_old', hist_len);
if all(isnan(D2_old)) && evalin('base','exist(''d2'',''var'')')
    D2_old = evalin('base','d2(:)');
end
D3_old = getVarOrNaN('D3_old', hist_len);
if all(isnan(D3_old)) && evalin('base','exist(''d3'',''var'')')
    D3_old = evalin('base','d3(:)');
end

%% ======= 兼容读取历史 CWSSI 并计算 WS_old（直接可用） =======

% 你脚本中历史长度（确保一致）
if ~exist('hist_len','var')
    hist_len = 12; % 若脚本中已定义请忽略
end

% 1) 尝试从若干常见变量名读取历史 CWSSI，优先级按数组顺序
candidate_names = {'CWSSI_old','cwssi','CWSSI','CW_old','CWSSI_hist','CWSSI_history'};
found = false;
for k = 1:length(candidate_names)
    nm = candidate_names{k};
    if evalin('base', ['exist(''' nm ''',''var'')'])
        tmp = evalin('base',[nm '(:)']); % 列向量
        tmp = tmp(:);
        % 截断或填充到 hist_len
        CWSSI_old = nan(hist_len,1);
        L = min(length(tmp), hist_len);
        CWSSI_old(1:L) = tmp(1:L);
        found = true;
        fprintf('Loaded CWSSI_old from workspace variable: %s (length %d -> normalized to %d)\n', nm, length(tmp), hist_len);
        break;
    end
end
if ~found
    CWSSI_old = nan(hist_len,1);
    warning('No CWSSI-like variable found in workspace. CWSSI_old set to NaNs (length %d).', hist_len);
end

% 3) 计算 WS_old（使用 min-max 归一化与固定权重）
wS1 = 0.52; wS2 = 0.48;
wD1 = 0.32; wD2 = 0.35; wD3 = 0.33;

% 只在至少有部分指标不是 NaN 时进行计算
if all(isnan([S1_old,S2_old,D1_old,D2_old,D3_old]), 'all')
    WS_old = nan(hist_len,1);
    warning('全部历史指标缺失，无法计算 WS_old.');
else
    % 计算历史 min/max（忽略 NaN）
    amin_hist = [nanmin(S1_old), nanmin(S2_old), nanmin(D1_old), nanmin(D2_old), nanmin(D3_old)];
    amax_hist = [nanmax(S1_old), nanmax(S2_old), nanmax(D1_old), nanmax(D2_old), nanmax(D3_old)];
    rng_hist = amax_hist - amin_hist;
    % 防止 0 范围
    idxZero = rng_hist==0;
    if any(idxZero)
        rng_hist(idxZero) = eps;
        warning('某些历史变量 max==min，已用 eps 代替差值以避免除零。');
    end

    % min-max 标准化（保留 NaN）
    S1n_old = (S1_old - amin_hist(1)) ./ rng_hist(1);
    S2n_old = (S2_old - amin_hist(2)) ./ rng_hist(2);
    D1n_old = (D1_old - amin_hist(3)) ./ rng_hist(3);
    D2n_old = (D2_old - amin_hist(4)) ./ rng_hist(4);
    D3n_old = (D3_old - amin_hist(5)) ./ rng_hist(5);

    % 组合
    S_total_old = wS1 .* S1n_old + wS2 .* S2n_old;
    D_total_old = wD1 .* D1n_old + wD2 .* D2n_old + wD3 .* D3n_old;

    % WS %
    WS_old = nan(hist_len,1);
    valid_mask = S_total_old ~= 0 & ~isnan(S_total_old) & ~isnan(D_total_old);
    WS_old(valid_mask) = (D_total_old(valid_mask) ./ S_total_old(valid_mask)) * 100;
    % 其余位置保持 NaN
end

% 将结果写回 base workspace（便于后续绘图直接使用）
assignin('base','CWSSI_old',CWSSI_old);
assignin('base','WS_old',WS_old);
assignin('base','S_total_old',S_total_old);
assignin('base','D_total_old',D_total_old);

fprintf('CWSSI_old 和 WS_old 已设置到 workspace（长度=%d）。\n', hist_len);


% prediction series
S1_pre = getVarOrNaN('S1_pre', future_len);
S2_pre = getVarOrNaN('S2_pre', future_len);
D1_pre = getVarOrNaN('D1_pre', future_len);
D2_pre = getVarOrNaN('D2_pre', future_len);
D3_pre = getVarOrNaN('D3_pre', future_len);
CWSSI_pre = getVarOrNaN('CWSSI_pre', future_len);
WS_pre = getVarOrNaN('WS_pre', future_len);

% intervention
CWSSI_int = getVarOrNaN('CWSSI_int', future_len);
WS_int = getVarOrNaN('WS_int', future_len);

% Inform about missing essentials (non-fatal)
if all(isnan(CWSSI_old))
    warning('CWSSI_old missing or all NaN.');
end
if all(isnan(CWSSI_pre))
    warning('CWSSI_pre missing or all NaN.');
end

% set default fonts
set(groot,'DefaultAxesFontName',fontname,'DefaultTextFontName',fontname);

%% =========================
% 1) CWSSI 模型框架图（流程框图）
%% =========================
fig1 = figure('Name','CWSSI_model_framework','NumberTitle','off','Units','pixels','Position',[100 100 1400 300]);
clf; axis off; hold on;
title('CWSSI Model Framework','FontName',fontname,'FontSize',16,'FontWeight','bold');

% layout
x0 = 0.02; w = 0.16; h = 0.35; gapx = 0.02; y = 0.3;
nodes = {
    'Data Input','2009-2020: S1/S2/D1/D2/D3';
    'Standardization','Extremum method:\nX''''=(X-X_{min})/(X_{max}-X_{min})\nNeg: X''''=(X_{max}-X)/(X_{max}-X_{min})';
    'PCA','Extract 2 PCs (cumulative var = 89.2%)\nCompute S_{total}, D_{total}';
    'Regression','CWSSI = 0.68*S_{total} + 0.32*D_{total} + \epsilon\nR^2 = 0.82';
    'GM(1,1)','GM(1,1) dynamic adj.\nReduce forecast error ≈30%';
    'CWSSI Output','Range 0-1 (higher = safer)'
};
n = size(nodes,1);
for i=1:n
    xi = x0 + (i-1)*(w+gapx);
    rectangle('Position',[xi,y,w,h],'FaceColor',[0.96 0.96 1],'EdgeColor','k','LineWidth',1.2);
    text(xi + w/2, y + h*0.65, nodes{i,1}, 'HorizontalAlignment','center','FontName',fontname,'FontSize',12,'FontWeight','bold');
    % Use interpreter none to avoid TeX issues for underscores
    text(xi + w/2, y + h*0.35, nodes{i,2}, 'HorizontalAlignment','center','FontName',fontname,'FontSize',10,'Interpreter','none');
    if i<n
        ax1 = xi + w; ax2 = xi + w + gapx;
        % annotation uses normalized figure coord, so convert
        % compute normalized positions
        ax = gca;
        annotation('arrow',[ax1 ax2],[y+h/2 y+h/2],'LineWidth',1.2);
    end
end

set(gcf,'PaperPositionMode','auto');
print(fig1,'CWSSI_model_framework.png','-dpng',sprintf('-r%d',save_dpi));
print(fig1,'CWSSI_model_framework.eps','-depsc2','-r300');

%% =========================
% 2) Drivers framework (双栏并列)
%% =========================
fig2 = figure('Name','Drivers_framework','NumberTitle','off','Units','pixels','Position',[100 100 900 600]);
clf; axis off; hold on;
title('Drivers of Water Scarcity in China','FontName',fontname,'FontSize',16,'FontWeight','bold');

% boxes
leftx = 0.05; colw = 0.43; colh = 0.6; ly = 0.18;
rectangle('Position',[leftx,ly,colw,colh],'EdgeColor','k','LineWidth',1.2);
text(leftx+colw/2,ly+colh-0.05,'Natural-type Water Scarcity','HorizontalAlignment','center','FontName',fontname,'FontSize',12,'FontWeight','bold');

% small bullet markers using plot (no toolbox)
plot(leftx+0.06, ly+colh-0.16, 'ko', 'MarkerFaceColor','k', 'MarkerSize',6);
text(leftx+0.12, ly+colh-0.16,'Climate: S1 2720–2840 ×10^9 m^3/year (N. China rainfall 400–600 mm)','FontName',fontname,'FontSize',10,'Interpreter','none');

plot(leftx+0.06, ly+colh-0.24, 'ko', 'MarkerFaceColor','k', 'MarkerSize',6);
text(leftx+0.12, ly+colh-0.24,'Geography: S1 loading = 0.82 (core supply indicator)','FontName',fontname,'FontSize',10,'Interpreter','none');

% right box
rightx = 0.52;
rectangle('Position',[rightx,ly,colw,colh],'EdgeColor','k','LineWidth',1.2);
text(rightx+colw/2,ly+colh-0.05,'Economic-type Water Scarcity','HorizontalAlignment','center','FontName',fontname,'FontSize',12,'FontWeight','bold');

plot(rightx+0.06, ly+colh-0.16, 'ko', 'MarkerFaceColor','k', 'MarkerSize',6);
text(rightx+0.12, ly+colh-0.16,'Population / Infrastructure: D1 65–78 m^3/capita/year (rural WW treatment <40%)','FontName',fontname,'FontSize',10,'Interpreter','none');

plot(rightx+0.06, ly+colh-0.24, 'ko', 'MarkerFaceColor','k', 'MarkerSize',6);
text(rightx+0.12, ly+colh-0.24,'Economy / Policy: D2 128–215 m^3/Million USD GDP; loading=0.71','FontName',fontname,'FontSize',10,'Interpreter','none');

% center arrow
annotation('arrow',[0.48 0.52],[0.66 0.66],'LineWidth',1.2);
text(0.5,0.69,'Drive →','HorizontalAlignment','center','FontName',fontname,'FontSize',11);

% vertical dashed separator
plot([0.495 0.495],[ly ly+colh+0.02],'--k');

set(gcf,'PaperPositionMode','auto');
print(fig2,'Drivers_framework.png','-dpng',sprintf('-r%d',save_dpi));
print(fig2,'Drivers_framework.eps','-depsc2','-r300');

%% =========================
% 3) 2041 预测流程图（分支并行）
%% =========================
fig3 = figure('Name','Forecast2041_flow','NumberTitle','off','Units','pixels','Position',[100 100 1000 600]);
clf; axis off; hold on;
title('2041 Water Situation Forecast Framework','FontName',fontname,'FontSize',16,'FontWeight','bold');

% input
rectangle('Position',[0.33,0.84,0.34,0.08],'EdgeColor','k','FaceColor',[0.96 0.96 1]);
text(0.5,0.88,'Input: Historical S1/S2/D1/D2/D3 (2009-2020)','HorizontalAlignment','center','FontName',fontname,'FontSize',11);

% left GM branch
rectangle('Position',[0.08,0.58,0.36,0.12],'EdgeColor','k','FaceColor',[0.9 0.95 1]);
text(0.08+0.36/2,0.64,'GM(1,1) prediction','HorizontalAlignment','center','FontName',fontname,'FontSize',11,'FontWeight','bold');
text(0.08+0.36/2,0.60,'Predict S1, S2, D2 (avg error 3.8%)','HorizontalAlignment','center','FontName',fontname,'FontSize',10);

% right regression branch
rectangle('Position',[0.54,0.58,0.36,0.12],'EdgeColor','k','FaceColor',[1 0.94 0.9]);
text(0.54+0.36/2,0.64,'Regression prediction','HorizontalAlignment','center','FontName',fontname,'FontSize',11,'FontWeight','bold');
text(0.54+0.36/2,0.60,'Predict D1, D3 (D1 = 0.5*UrbanRate + 45; R^2 = 0.78)','HorizontalAlignment','center','FontName',fontname,'FontSize',10);

% arrows (color-coded)
annotation('arrow',[0.5 0.26],[0.78 0.68],'Color',[0 0 1],'LineWidth',1.2);
annotation('arrow',[0.5 0.74],[0.78 0.68],'Color',[1 0 0],'LineWidth',1.2);

% aggregation box
rectangle('Position',[0.30,0.36,0.40,0.12],'EdgeColor','k','FaceColor',[0.95 1 0.95]);
text(0.3+0.4/2,0.41,'Aggregate: Compute S_{total}, D_{total}','HorizontalAlignment','center','FontName',fontname,'FontSize',11);
text(0.3+0.4/2,0.37,'S_{total}=0.52*S1'''' +0.48*S2''''; D_{total}=0.32*D1''''+0.35*D2''''+0.33*D3''''','HorizontalAlignment','center','FontName',fontname,'FontSize',9,'Interpreter','none');

% arrow to output
annotation('arrow',[0.5 0.5],[0.36 0.26],'LineWidth',1.2);

% output box
rectangle('Position',[0.34,0.08,0.32,0.12],'EdgeColor','k','FaceColor',[1 1 0.9]);
text(0.34+0.32/2,0.14,'Output: 2041 CWSSI & WS','HorizontalAlignment','center','FontName',fontname,'FontSize',11);
text(0.34+0.32/2,0.10,'WS = TotalWithdrawal / S1 × 100%','HorizontalAlignment','center','FontName',fontname,'FontSize',9,'Interpreter','none');

set(gcf,'PaperPositionMode','auto');
print(fig3,'Forecast2041_flow.png','-dpng',sprintf('-r%d',save_dpi));
print(fig3,'Forecast2041_flow.eps','-depsc2','-r300');

%% =========================
% 4) 2009-2041 CWSSI + 指标趋势图（多轴折线）
%% =========================
% Compose years for plotting
year_plot = [hist_years; pred_years];

% combine CWSSI
cw_all = nan(length(year_plot),1);
if ~all(isnan(CWSSI_old))
    Lh = min(length(CWSSI_old), length(hist_years));
    cw_all(1:Lh) = CWSSI_old(1:Lh);
end
if ~all(isnan(CWSSI_pre))
    Lp = min(length(CWSSI_pre), length(pred_years));
    cw_all(length(hist_years)+(1:Lp)) = CWSSI_pre(1:Lp);
end

% combine indicators using helper
S1_all = combine_series(S1_old, S1_pre, hist_years, pred_years);
S2_all = combine_series(S2_old, S2_pre, hist_years, pred_years);
D1_all = combine_series(D1_old, D1_pre, hist_years, pred_years);
D2_all = combine_series(D2_old, D2_pre, hist_years, pred_years);
D3_all = combine_series(D3_old, D3_pre, hist_years, pred_years);

fig4 = figure('Name','Trends_2009_2041','NumberTitle','off','Units','pixels','Position',[100 100 1000 600]);
clf;

yyaxis left;
hCW = plot(year_plot, cw_all, '-ok','LineWidth',2,'MarkerFaceColor','k'); hold on;
ylabel('CWSSI (0-1)','FontName',fontname,'FontSize',label_fontsize);
ylim([0 1]); yticks(0:0.1:1);

yyaxis right;
p1 = plot(year_plot, S1_all, '-','LineWidth',1.5); hold on;
p2 = plot(year_plot, S2_all, '--','LineWidth',1.2);
p3 = plot(year_plot, D1_all, '-','LineWidth',1.2);
p4 = plot(year_plot, D2_all, '--','LineWidth',1.2);
p5 = plot(year_plot, D3_all, '-','LineWidth',1.2);

% set colors per spec
set(p1,'Color',[0 0.4470 0.7410]);
set(p2,'Color',[0 0.4470 0.7410]); set(p2,'LineStyle','--');
set(p3,'Color',[0.85 0.33 0.1]);
set(p4,'Color',[0.85 0.33 0.1]); set(p4,'LineStyle','--');
set(p5,'Color',[0.9290 0.6940 0.1250]);

ylabel('Indicators (units in legend)','FontName',fontname,'FontSize',label_fontsize);

% xticks as requested if within range
xticks_req = [2009,2015,2020,2025,2030,2035,2041];
xticks_use = xticks_req(xticks_req >= year_plot(1) & xticks_req <= year_plot(end));
set(gca,'XTick',xticks_use,'FontName',fontname);

legend({'CWSSI','S1 (10^9 m^3/year)','S2 (10^9 m^3/year)','D1 (m^3/capita/year)',...
    'D2 (m^3/Million USD GDP)','D3 (capita·year/m^3)'},'Location','northeast','FontName',fontname);

xlabel('Year','FontName',fontname);
title('CWSSI and Indicators: Historical + Forecast','FontName',fontname,'FontSize',title_fontsize);

% annotate last historical & final forecast if not NaN
last_hist = hist_years(end);
if ~all(isnan(cw_all(1:length(hist_years))))
    idx_hist = find(year_plot==last_hist,1);
    if ~isempty(idx_hist)
        text(last_hist, cw_all(idx_hist)+0.06, sprintf('%d CWSSI=%.3f', last_hist, cw_all(idx_hist)), 'FontName',fontname,'FontSize',anno_fontsize);
    end
end
if ~isnan(cw_all(end))
    text(year_plot(end), cw_all(end)+0.06, sprintf('%d CWSSI=%.3f', year_plot(end), cw_all(end)), 'FontName',fontname,'FontSize',anno_fontsize);
end

grid on;
set(gcf,'PaperPositionMode','auto');
print(fig4,'Trends_2009_2041.png','-dpng',sprintf('-r%d',save_dpi));
print(fig4,'Trends_2009_2041.eps','-depsc2','-r300');

%% =========================
% 5) 干预方案效果对比图（柱状 + 折线 WS）
%% =========================
% baseline value: prefer CWSSI_actual_2023 if exists, else use last historical
if evalin('base','exist(''CWSSI_actual_2023'',''var'')')
    baseline_year = 2023;
    baseline_val = evalin('base','CWSSI_actual_2023');
else
    baseline_year = hist_years(end);
    baseline_val = CWSSI_old(end);
end

% get 2041 no-intervention value
idx2041 = find(pred_years==2041,1);
if ~isempty(idx2041) && idx2041 <= length(CWSSI_pre) && ~isnan(CWSSI_pre(idx2041))
    val_2041_no = CWSSI_pre(idx2041);
    ws_2041_no = WS_pre(idx2041);
else
    val_2041_no = CWSSI_pre(end);
    ws_2041_no = WS_pre(end);
end
% intervention value (user-provided or simulate)
if ~all(isnan(CWSSI_int))
    if ~isempty(idx2041) && idx2041 <= length(CWSSI_int)
        val_2041_int = CWSSI_int(idx2041);
    else
        val_2041_int = CWSSI_int(end);
    end
else
    val_2041_int = NaN;
end

if ~all(isnan(WS_int))
    if ~isempty(idx2041) && idx2041 <= length(WS_int)
        ws_2041_int = WS_int(idx2041);
    else
        ws_2041_int = WS_int(end);
    end
else
    ws_2041_int = NaN;
end


%% ---------------- FIXED PART BELOW ----------------
% Build unified future indicator vector (avoid dimension mismatch)
future_all = [S1_pre(:); S2_pre(:); D1_pre(:); D2_pre(:); D3_pre(:)];

% If no explicit intervention provided, simulate example intervention
if isnan(val_2041_int) && any(~isnan(future_all))

    % use last available predicted indicators as baseline
    S1_f = S1_pre(find(~isnan(S1_pre),1,'last'));
    S2_f = S2_pre(find(~isnan(S2_pre),1,'last'));
    D1_f = D1_pre(find(~isnan(D1_pre),1,'last'));
    D2_f = D2_pre(find(~isnan(D2_pre),1,'last'));
    D3_f = D3_pre(find(~isnan(D3_pre),1,'last'));

    if isempty(S1_f), S1_f = nanmean(S1_pre); end
    if isempty(S2_f), S2_f = nanmean(S2_pre); end
    if isempty(D1_f), D1_f = nanmean(D1_pre); end
    if isempty(D2_f), D2_f = nanmean(D2_pre); end
    if isempty(D3_f), D3_f = nanmean(D3_pre); end

    % Intervention changes
    S2_f_i = S2_f * 1.40;
    D1_f_i = D1_f * 0.85;
    D2_f_i = D2_f * 0.70;

    % approximate normalization
    s1n = normalize_simple(S1_f, S1_old);
    s2n = normalize_simple(S2_f, S2_old);
    d1n = normalize_simple(D1_f, D1_old);
    d2n = normalize_simple(D2_f, D2_old);
    d3n = normalize_simple(D3_f, D3_old);

    s2n_i = normalize_simple(S2_f_i, S2_old);
    d1n_i = normalize_simple(D1_f_i, D1_old);
    d2n_i = normalize_simple(D2_f_i, D2_old);

    % compose S_total and D_total
    S_total_no = 0.52*s1n + 0.48*s2n;
    D_total_no = 0.32*d1n + 0.35*d2n + 0.33*d3n;

    S_total_i  = 0.52*s1n + 0.48*s2n_i;
    D_total_i  = 0.32*d1n_i + 0.35*d2n_i + 0.33*d3n;

    % predict CWSSI via regression if available
    if evalin('base','exist(''B_ols'',''var'')')
        B_ols_local = evalin('base','B_ols');
        if numel(B_ols_local) >= 3
            val_2041_no  = [S_total_no, D_total_no, 1] * B_ols_local;
            val_2041_int = [S_total_i,  D_total_i,  1] * B_ols_local;
        else
            val_2041_no  = NaN;
            val_2041_int = NaN;
        end
    else
        % fallback
        if exist('val_2041_no','var') && ~isnan(val_2041_no)
            rel = (D_total_i - D_total_no) / max(abs(D_total_no), eps);
            val_2041_int = val_2041_no - 0.5 * rel;
        end
    end

    ws_2041_no  = (D_total_no ./ S_total_no) * 100;
    ws_2041_int = (D_total_i ./ S_total_i) * 100;

end

% Prepare plot
fig5 = figure('Name','Intervention_Comparison','NumberTitle','off','Units','pixels','Position',[100 100 900 500]);
clf;
ax = axes('Position',[0.08 0.12 0.65 0.75]); hold on;

bar_vals = [baseline_val, val_2041_no, val_2041_int];
b = bar(1:3, bar_vals, 0.6,'LineWidth',1);
b.FaceColor = 'flat';
b.CData(1,:) = [0.6 0.6 0.6];
b.CData(2,:) = [0 0.4470 0.7410];
b.CData(3,:) = [0.2 0.7 0.2];

ylim([0 0.8]);
ylabel('CWSSI','FontName',fontname);

% annotate bars
for i=1:3
    if ~isnan(bar_vals(i))
        text(i, bar_vals(i)+0.03, sprintf('%.3f', bar_vals(i)), 'HorizontalAlignment','center','FontName',fontname,'FontSize',10);
    end
end

% add WS curve on right axis (overlay)
ax2 = axes('Position',get(ax,'Position'),'Color','none','XAxisLocation','top','YAxisLocation','right','XTick',[]); hold on;
ws_vals = [NaN, ws_2041_no, ws_2041_int];
if ~all(isnan(ws_vals))
    plot(2:3, ws_vals(2:3), '-o','LineWidth',2,'MarkerFaceColor',[0.85 0.33 0.1],'Parent',ax2);
    set(ax2,'YColor',[0.85 0.33 0.1],'FontName',fontname);
    ylabel(ax2,'WS (%)','FontName',fontname);
    % choose reasonable ylim
    ylims = get(ax2,'YLim');
    set(ax2,'YLim',[20 40]);
end

set(ax,'XTick',1:3,'XTickLabel',{sprintf('%d (actual)',baseline_year),'2041 (no int)','2041 (with int)'},'FontName',fontname);
legend({'CWSSI'},'Location','northeast','FontName',fontname);

% annotate percent change on green bar
if ~isnan(bar_vals(2)) && ~isnan(bar_vals(3)) && bar_vals(2)~=0
    pct = (bar_vals(3)-bar_vals(2))/abs(bar_vals(2))*100;
    text(3,bar_vals(3)+0.05,sprintf('+%.1f%%',pct),'HorizontalAlignment','center','FontName',fontname,'FontSize',10);
end
% annotate WS percent change
if ~isnan(ws_vals(2)) && ~isnan(ws_vals(3)) && ws_vals(2)~=0
    pct_ws = (ws_vals(3)-ws_vals(2))/abs(ws_vals(2))*100;
    text(3, ws_vals(3)+1.0, sprintf('%.1f%%', pct_ws),'HorizontalAlignment','center','FontName',fontname,'FontSize',10,'Color',[0.85 0.33 0.1]);
end

title('Intervention Effect: CWSSI & WS','FontName',fontname);

set(gcf,'PaperPositionMode','auto');
print(fig5,'Intervention_Comparison.png','-dpng',sprintf('-r%d',save_dpi));
print(fig5,'Intervention_Comparison.eps','-depsc2','-r300');

disp('All figures generated and saved to current folder.');

end

%% =========================
% Helper functions
%% =========================

function out = safe_get(varname, len)
% read var from base workspace, return column vector of length len or NaNs
    if evalin('base',['exist(''' varname ''',''var'')'])
        tmp = evalin('base',[varname '(:)']);
        tmp = tmp(:);
        if length(tmp) >= len
            out = tmp(1:len);
        else
            out = nan(len,1);
            out(1:length(tmp)) = tmp;
        end
    else
        out = nan(len,1);
    end
end

function combined = combine_series(oldS, preS, hist_years, pred_years)
% Combine historical and predicted series safely into a single column vector
    hist_len = length(hist_years);
    fut_len = length(pred_years);
    combined = nan(hist_len + fut_len, 1);
    if ~isempty(oldS) && ~all(isnan(oldS))
        Lh = min(length(oldS), hist_len);
        combined(1:Lh) = oldS(1:Lh);
    end
    if ~isempty(preS) && ~all(isnan(preS))
        Lp = min(length(preS), fut_len);
        combined(hist_len + (1:Lp)) = preS(1:Lp);
    end
end

function v = normalize_simple(val, histvec)
% simple min-max normalize a scalar val using historical vector histvec
    if isempty(histvec) || all(isnan(histvec))
        v = NaN;
        return;
    end
    amin = nanmin(histvec(:));
    amax = nanmax(histvec(:));
    if amax - amin == 0
        v = 0.5; % arbitrary
    else
        v = (val - amin) / (amax - amin);
    end
end
