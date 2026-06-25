import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义24小时时间段
shi_jian_duan = {f'{i:02d}:00-{i+1:02d}:00': (i, i+1) for i in range(24)}

# 1. 数据加载
print("加载数据...")
try:
    data = pd.read_csv('Attachment 1.csv')
except FileNotFoundError:
    print("错误：未找到 Attachment 1.csv 文件！")
    exit()

data.columns = ['yong_hu_ID', 'yong_hu_xing_wei', 'bo_zhu_ID', 'shi_jian']
data['shi_jian'] = pd.to_datetime(data['shi_jian'])
data = data[(data['shi_jian'] >= '2024-07-11') & (data['shi_jian'] <= '2024-07-20')]
print(f"数据加载完成！共 {len(data)} 条记录")

# 筛选目标用户
mu_biao_yong_hu = ['U10', 'U1951', 'U1833', 'U26447']
data = data[data['yong_hu_ID'].isin(mu_biao_yong_hu)]
print(f"筛选目标用户后：共 {len(data)} 条记录")

# 2. 特征工程
print("执行特征工程...")
data['xiao_shi'] = data['shi_jian'].dt.hour
data['ri_qi'] = data['shi_jian'].dt.day
data['shi_jian_cha'] = data.groupby('yong_hu_ID')['shi_jian'].diff().dt.total_seconds().fillna(0) / 3600
yong_hu_zai_xian_pin_lv = data.groupby('yong_hu_ID').apply(
    lambda x: len(x) / ((x['shi_jian'].max() - x['shi_jian'].min()).total_seconds() / 86400 + 1e-10)
).reset_index(name='zai_xian_pin_lv')
data = data.merge(yong_hu_zai_xian_pin_lv, on='yong_hu_ID')
data['shang_ci_huo_dong_shi_jian'] = data.groupby('yong_hu_ID')['shi_jian'].diff(-1).shift(1).dt.total_seconds().fillna(0) / 3600
data['shang_ci_huo_dong_shi_jian'] = data['shang_ci_huo_dong_shi_jian'].clip(lower=0)
def fen_pei_shi_jian_duan(xiao_shi):
    for duan, (kai_shi, jie_shu) in shi_jian_duan.items():
        if kai_shi <= xiao_shi < jie_shu:
            return duan
    return '23:00-24:00'
data['shi_jian_duan'] = data['xiao_shi'].apply(fen_pei_shi_jian_duan)
shi_jian_duan_ya_bian_liang = pd.get_dummies(data['shi_jian_duan'], prefix='shi_jian_duan')
data = pd.concat([data, shi_jian_duan_ya_bian_liang], axis=1)
for duan in shi_jian_duan:
    lie = f'shi_jian_duan_{duan}'
    if lie not in data.columns:
        data[lie] = 0
data['ju_kai_shi_shi_jian'] = (data['shi_jian'] - pd.to_datetime('2024-07-11')).dt.total_seconds() / 86400
yong_hu_huo_dong = data.groupby('yong_hu_ID').apply(
    lambda x: x[x['shi_jian'] >= x['shi_jian'].max() - pd.Timedelta(days=3)]['yong_hu_xing_wei'].count() / 3
).reset_index(name='jin_qi_huo_dong')
data = data.merge(yong_hu_huo_dong, on='yong_hu_ID')
yong_hu_tong_ji = data.groupby('yong_hu_ID').agg({
    'yong_hu_xing_wei': ['count', lambda x: (x == 2).sum(), lambda x: (x == 3).sum(), lambda x: (x == 4).sum()]
}).reset_index()
yong_hu_tong_ji.columns = ['yong_hu_ID', 'xing_wei_ji_shu', 'dian_zan_ji_shu', 'ping_lun_ji_shu', 'guan_zhu_ji_shu']
data = data.merge(yong_hu_tong_ji, on='yong_hu_ID')
bo_zhu_tong_ji = data.groupby('bo_zhu_ID').agg({
    'yong_hu_xing_wei': ['count', lambda x: (x == 2).sum(), lambda x: (x == 3).sum(), lambda x: (x == 4).sum()]
}).reset_index()
bo_zhu_tong_ji.columns = ['bo_zhu_ID', 'bo_zhu_xing_wei_ji_shu', 'bo_zhu_dian_zan_ji_shu', 'bo_zhu_ping_lun_ji_shu', 'bo_zhu_guan_zhu_ji_shu']
data = data.merge(bo_zhu_tong_ji, on='bo_zhu_ID')
yong_hu_bo_zhu_hu_dong = data.groupby(['yong_hu_ID', 'bo_zhu_ID'])['yong_hu_xing_wei'].count().reset_index()
yong_hu_bo_zhu_hu_dong.columns = ['yong_hu_ID', 'bo_zhu_ID', 'yong_hu_bo_zhu_hu_dong_ji_shu']
data = data.merge(yong_hu_bo_zhu_hu_dong, on=['yong_hu_ID', 'bo_zhu_ID'], how='left')
data['yong_hu_bo_zhu_hu_dong_ji_shu'] = data['yong_hu_bo_zhu_hu_dong_ji_shu'].fillna(0)
yong_hu_bian_ma_qi = {u: i for i, u in enumerate(mu_biao_yong_hu)}
bo_zhu_ID = data['bo_zhu_ID'].unique()
bo_zhu_bian_ma_qi = {b: i for i, b in enumerate(bo_zhu_ID)}
data['yong_hu_ID_bian_ma'] = data['yong_hu_ID'].map(yong_hu_bian_ma_qi)
data['bo_zhu_ID_bian_ma'] = data['bo_zhu_ID'].map(bo_zhu_bian_ma_qi)
xing_wei_ya_bian_liang = pd.get_dummies(data['yong_hu_xing_wei'], prefix='xing_wei')
data = pd.concat([data, xing_wei_ya_bian_liang], axis=1)
for i in range(1, 5):
    lie = f'xing_wei_{i}'
    if lie not in data.columns:
        data[lie] = 0

# 构造时间序列
xu_lie_chang_du = 3
te_zheng = ['yong_hu_ID_bian_ma', 'bo_zhu_ID_bian_ma', 'xiao_shi', 'ri_qi', 'shi_jian_cha', 'zai_xian_pin_lv', 'shang_ci_huo_dong_shi_jian',
            'jin_qi_huo_dong', 'xing_wei_ji_shu', 'dian_zan_ji_shu', 'ping_lun_ji_shu', 'guan_zhu_ji_shu',
            'bo_zhu_xing_wei_ji_shu', 'bo_zhu_dian_zan_ji_shu', 'bo_zhu_ping_lun_ji_shu', 'bo_zhu_guan_zhu_ji_shu',
            'yong_hu_bo_zhu_hu_dong_ji_shu', 'xing_wei_1', 'xing_wei_2', 'xing_wei_3', 'xing_wei_4'] + \
           [f'shi_jian_duan_{duan}' for duan in shi_jian_duan]
X_xu_lie = []
y_zai_xian = []
y_hu_dong = []
for yong_hu in tqdm(mu_biao_yong_hu, desc="构造序列"):
    yong_hu_shu_ju = data[data['yong_hu_ID'] == yong_hu].sort_values('shi_jian')
    print(f"用户 {yong_hu} 数据：{len(yong_hu_shu_ju)} 条记录")
    if len(yong_hu_shu_ju) < xu_lie_chang_du:
        print(f"用户 {yong_hu} 数据不足（< {xu_lie_chang_du} 条记录），跳过")
        continue
    for i in range(len(yong_hu_shu_ju) - xu_lie_chang_du):
        xu_lie = yong_hu_shu_ju[te_zheng].iloc[i:i+xu_lie_chang_du].values
        dang_qian_shi_jian = yong_hu_shu_ju.iloc[i+xu_lie_chang_du-1]['shi_jian']
        zai_xian_gai_lv = 1.0 if len(yong_hu_shu_ju[yong_hu_shu_ju['shi_jian'].dt.date == dang_qian_shi_jian.date()]) > 0 else 0.0
        hu_dong = yong_hu_shu_ju.iloc[i+xu_lie_chang_du-1][['xing_wei_2', 'xing_wei_3', 'xing_wei_4']].sum()
        X_xu_lie.append(xu_lie)
        y_zai_xian.append(zai_xian_gai_lv)
        y_hu_dong.append(hu_dong)
if len(X_xu_lie) == 0:
    print("错误：未构造任何序列！")
    exit()
X_xu_lie = np.array(X_xu_lie)
y_zai_xian = np.array(y_zai_xian)
y_hu_dong = np.array(y_hu_dong)
y_hu_dong = np.clip(y_hu_dong, 0, 5)
print(f"序列构造完成：共 {len(X_xu_lie)} 个序列")
biao_zhun_hua_qi = StandardScaler()
X_xu_lie = biao_zhun_hua_qi.fit_transform(X_xu_lie.reshape(-1, X_xu_lie.shape[-1])).reshape(X_xu_lie.shape)
X_xun_lian, X_ce_shi, y_zai_xian_xun_lian, y_zai_xian_ce_shi, y_hu_dong_xun_lian, y_hu_dong_ce_shi = train_test_split(
    X_xu_lie, y_zai_xian, y_hu_dong, test_size=0.2, random_state=42
)
print("数据预处理完成！")

# 3. 定义GRU模型
class GRU_mo_xing(nn.Module):
    def __init__(self, shu_ru_da_xiao, yin_cang_da_xiao, ceng_shu, diu_qi_lv=0.3):
        super(GRU_mo_xing, self).__init__()
        self.gru = nn.GRU(shu_ru_da_xiao, yin_cang_da_xiao, ceng_shu, batch_first=True, dropout=diu_qi_lv if ceng_shu > 1 else 0)
        self.bn = nn.BatchNorm1d(yin_cang_da_xiao)
        self.fc_zai_xian = nn.Linear(yin_cang_da_xiao, 1)
        self.fc_hu_dong = nn.Linear(yin_cang_da_xiao, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        gru_shu_chu, _ = self.gru(x)
        gru_shu_chu = gru_shu_chu[:, -1, :]
        gru_shu_chu = self.bn(gru_shu_chu)
        zai_xian_yu_ce = self.sigmoid(self.fc_zai_xian(gru_shu_chu))
        hu_dong_yu_ce = self.fc_hu_dong(gru_shu_chu)
        return zai_xian_yu_ce, hu_dong_yu_ce
shu_ru_da_xiao = X_xu_lie.shape[-1]
yin_cang_da_xiao = 16
ceng_shu = 1
mo_xing = GRU_mo_xing(shu_ru_da_xiao, yin_cang_da_xiao, ceng_shu)
she_bei = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mo_xing = mo_xing.to(she_bei)
zai_xian_sun_shi_han_shu = nn.BCELoss()
hu_dong_sun_shi_han_shu = nn.MSELoss()
you_hua_qi = optim.Adam(mo_xing.parameters(), lr=0.001, weight_decay=1e-4)
diao_du_qi = optim.lr_scheduler.ReduceLROnPlateau(you_hua_qi, 'min', patience=5, factor=0.5)

# 4. 模型训练
print("开始模型训练...")
lun_shu = 250
xun_lian_sun_shi = []
ce_shi_sun_shi = []
zui_jia_sun_shi = float('inf')
nai_xin = 20
zao_ting_ji_shu = 0
mo_xing.eval()
with torch.no_grad():
    X_ce_shi_zhang_liang = torch.tensor(X_ce_shi, dtype=torch.float32).to(she_bei)
    zai_xian_yu_ce, _ = mo_xing(X_ce_shi_zhang_liang)
    zai_xian_yu_ce = zai_xian_yu_ce.cpu().numpy().squeeze()
    yu_zhi = np.arange(0.1, 1.0, 0.1)
    f1_fen_shu = [f1_score(y_zai_xian_ce_shi, zai_xian_yu_ce > t) for t in yu_zhi]
    zui_jia_yu_zhi = yu_zhi[np.argmax(f1_fen_shu)]
    print(f"最佳在线预测阈值：{zui_jia_yu_zhi:.2f}")
for lun in tqdm(range(lun_shu), desc="训练进度"):
    mo_xing.train()
    lun_sun_shi = 0
    for i in range(0, len(X_xun_lian), 16):
        X_pi_ci = torch.tensor(X_xun_lian[i:i+16], dtype=torch.float32).to(she_bei)
        y_zai_xian_pi_ci = torch.tensor(y_zai_xian_xun_lian[i:i+16], dtype=torch.float32).to(she_bei)
        y_hu_dong_pi_ci = torch.tensor(y_hu_dong_xun_lian[i:i+16], dtype=torch.float32).to(she_bei)
        you_hua_qi.zero_grad()
        zai_xian_yu_ce, hu_dong_yu_ce = mo_xing(X_pi_ci)
        zai_xian_sun_shi = zai_xian_sun_shi_han_shu(zai_xian_yu_ce.squeeze(), y_zai_xian_pi_ci)
        zhong_liang = torch.where(y_hu_dong_pi_ci > 0, 2.0, 1.0).to(she_bei)
        hu_dong_sun_shi = (hu_dong_sun_shi_han_shu(hu_dong_yu_ce.squeeze(), y_hu_dong_pi_ci) * zhong_liang).mean()
        sun_shi = 0.5 * zai_xian_sun_shi + 0.5 * hu_dong_sun_shi
        sun_shi.backward()
        you_hua_qi.step()
        lun_sun_shi += sun_shi.item()
    xun_lian_sun_shi.append(lun_sun_shi / (len(X_xun_lian) // 16 + 1))
    mo_xing.eval()
    with torch.no_grad():
        X_ce_shi_zhang_liang = torch.tensor(X_ce_shi, dtype=torch.float32).to(she_bei)
        y_zai_xian_ce_shi_zhang_liang = torch.tensor(y_zai_xian_ce_shi, dtype=torch.float32).to(she_bei)
        y_hu_dong_ce_shi_zhang_liang = torch.tensor(y_hu_dong_ce_shi, dtype=torch.float32).to(she_bei)
        zai_xian_yu_ce, hu_dong_yu_ce = mo_xing(X_ce_shi_zhang_liang)
        zai_xian_sun_shi = zai_xian_sun_shi_han_shu(zai_xian_yu_ce.squeeze(), y_zai_xian_ce_shi_zhang_liang)
        hu_dong_sun_shi = hu_dong_sun_shi_han_shu(hu_dong_yu_ce.squeeze(), y_hu_dong_ce_shi_zhang_liang)
        ce_shi_sun_shi_zhi = 0.5 * zai_xian_sun_shi + 0.5 * hu_dong_sun_shi
        ce_shi_sun_shi.append(ce_shi_sun_shi_zhi.item())
    diao_du_qi.step(ce_shi_sun_shi_zhi)
    if ce_shi_sun_shi_zhi < zui_jia_sun_shi:
        zui_jia_sun_shi = ce_shi_sun_shi_zhi
        zao_ting_ji_shu = 0
        torch.save(mo_xing.state_dict(), 'zui_jia_mo_xing_20240723.pth')
    else:
        zao_ting_ji_shu += 1
        if zao_ting_ji_shu >= nai_xin:
            print("触发早停")
            break
print("模型训练完成！")
plt.figure(figsize=(10, 6))
plt.plot(xun_lian_sun_shi, label='训练损失', color='blue')
plt.plot(ce_shi_sun_shi, label='测试损失', color='orange')
plt.title('训练和测试损失曲线')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()
plt.grid(True)
plt.savefig('sun_shi_qu_xian.png')
plt.close()
mo_xing.load_state_dict(torch.load('zui_jia_mo_xing_20240723.pth'))

# 5. 模型评估
print("评估模型...")
mo_xing.eval()
with torch.no_grad():
    X_ce_shi_zhang_liang = torch.tensor(X_ce_shi, dtype=torch.float32).to(she_bei)
    zai_xian_yu_ce, hu_dong_yu_ce = mo_xing(X_ce_shi_zhang_liang)
    zai_xian_yu_ce = zai_xian_yu_ce.cpu().numpy().squeeze()
    hu_dong_yu_ce = hu_dong_yu_ce.cpu().numpy().squeeze()
auc = roc_auc_score(y_zai_xian_ce_shi, zai_xian_yu_ce) if len(np.unique(y_zai_xian_ce_shi)) > 1 else 0.0
mre = np.mean(np.abs(hu_dong_yu_ce - y_hu_dong_ce_shi) / (y_hu_dong_ce_shi + 1e-10))
mare = np.mean(np.abs(hu_dong_yu_ce - y_hu_dong_ce_shi) / (np.abs(y_hu_dong_ce_shi) + 1e-10))
r2 = r2_score(y_hu_dong_ce_shi, hu_dong_yu_ce)
mae = mean_absolute_error(y_hu_dong_ce_shi, hu_dong_yu_ce)
mse = mean_squared_error(y_hu_dong_ce_shi, hu_dong_yu_ce)
ping_gu_shu_ju_kuang = pd.DataFrame({
    'zhi_biao': ['zai_xian_AUC', 'hu_dong_MRE', 'hu_dong_MARE', 'hu_dong_R²', 'hu_dong_MAE', 'hu_dong_MSE'],
    'zhi': [auc * 100, mre * 100, mare * 100, r2 * 100, mae * 100, mse * 100]
})
print("\n模型评估结果（%）：")
print(ping_gu_shu_ju_kuang)
ping_gu_shu_ju_kuang.to_csv('ping_gu_zhi_biao_20240723.csv', index=False)

# 6. 预测2024年7月23日的用户行为
print("预测2024年7月23日的用户行为...")
jie_guo = []
yu_ce_ri_qi = pd.to_datetime('2024-07-23')
for yong_hu in tqdm(mu_biao_yong_hu, desc="预测进度"):
    yong_hu_shu_ju = data[data['yong_hu_ID'] == yong_hu].sort_values('shi_jian')
    print(f"用户 {yong_hu} 数据：{len(yong_hu_shu_ju)} 条记录")
    if len(yong_hu_shu_ju) < xu_lie_chang_du:
        print(f"用户 {yong_hu} 数据不足，预测为离线")
        jie_guo.append([yong_hu, 0, '', 0, '', '', 0, '', '', 0, ''])
        continue
    # 确定用户在线时间段
    yong_hu_zai_xian_shi_duan = yong_hu_shu_ju['shi_jian_duan'].unique()
    if len(yong_hu_zai_xian_shi_duan) == 0:
        print(f"用户 {yong_hu} 无活跃时间段，使用所有时间段")
        yong_hu_zai_xian_shi_duan = list(shi_jian_duan.keys())
    print(f"用户 {yong_hu} 在线时间段：{yong_hu_zai_xian_shi_duan}")
    xu_lie = yong_hu_shu_ju[te_zheng].iloc[-xu_lie_chang_du:].values
    xu_lie = biao_zhun_hua_qi.transform(xu_lie.reshape(-1, xu_lie.shape[-1])).reshape(1, xu_lie_chang_du, -1)
    xu_lie_zhang_liang = torch.tensor(xu_lie, dtype=torch.float32).to(she_bei)
    mo_xing.eval()
    with torch.no_grad():
        zai_xian_yu_ce, hu_dong_yu_ce = mo_xing(xu_lie_zhang_liang)
        zai_xian_gai_lv = zai_xian_yu_ce.cpu().numpy().squeeze()
        hu_dong_yu_ce = hu_dong_yu_ce.cpu().numpy().squeeze()
    print(f"用户 {yong_hu} 在线概率：{zai_xian_gai_lv:.4f}")
    if zai_xian_gai_lv > zui_jia_yu_zhi:
        hu_dong_de_fen = []
        # Step 1: 为每个在线时间段和博主预测互动数
        for duan in yong_hu_zai_xian_shi_duan:
            for bo_zhu in bo_zhu_ID[:50]:  # 限制为前100个博主以提高效率
                # 创建新的序列副本
                duan_xu_lie = xu_lie.copy()
                # 重置所有时间段独热编码
                for seg in shi_jian_duan:
                    duan_xu_lie[:, :, te_zheng.index(f'shi_jian_duan_{seg}')] = 1 if seg == duan else 0
                duan_xiao_shi = shi_jian_duan[duan][0]
                duan_xu_lie[:, :, te_zheng.index('xiao_shi')] = duan_xiao_shi
                duan_xu_lie[:, :, te_zheng.index('ri_qi')] = 23
                duan_xu_lie[:, :, te_zheng.index('bo_zhu_ID_bian_ma')] = bo_zhu_bian_ma_qi.get(bo_zhu, 0)
                duan_xu_lie_zhang_liang = torch.tensor(duan_xu_lie, dtype=torch.float32).to(she_bei)
                with torch.no_grad():
                    _, hu_dong_yu_ce_duan = mo_xing(duan_xu_lie_zhang_liang)
                hu_dong = max(0, hu_dong_yu_ce_duan.cpu().numpy().squeeze().item())
                hu_dong_de_fen.append((bo_zhu, hu_dong, duan))
            # 调试：打印时间段的互动数分布
            shi_duan_hu_dong = [score for _, score, d in hu_dong_de_fen if d == duan]
            if shi_duan_hu_dong:
                print(f"用户 {yong_hu} 时间段 {duan} 互动数分布：均值={np.mean(shi_duan_hu_dong):.4f}, 标准差={np.std(shi_duan_hu_dong):.4f}")
        # Step 2: 按互动数排序，选择前3
        hu_dong_de_fen.sort(key=lambda x: x[1], reverse=True)
        qian_san_hu_dong = hu_dong_de_fen[:3]
        # Step 3: 构造结果
        jie_guo_hang = [yong_hu, 1]
        print(f"用户 {yong_hu} 互动数最高的前3位博主：")
        for i, (bo_zhu, hu_dong, duan) in enumerate(qian_san_hu_dong):
            jie_guo_hang.extend([bo_zhu, round(hu_dong, 2), duan])
            print(f"  博主 {bo_zhu}: 互动数 {hu_dong:.2f}, 时间段 {duan}")
        for i in range(len(qian_san_hu_dong), 3):
            jie_guo_hang.extend(['', 0, ''])
        jie_guo.append(jie_guo_hang)
    else:
        jie_guo.append([yong_hu, 0, '', 0, '', '', 0, '', '', 0, ''])
jie_guo_shu_ju_kuang = pd.DataFrame(jie_guo, columns=[
    'yong_hu_ID', 'shi_fou_zai_xian (1=是, 0=否)',
    'qian_yi_bo_zhu', 'hu_dong1', 'shi_jian_duan1',
    'qian_er_bo_zhu', 'hu_dong2', 'shi_jian_duan2',
    'qian_san_bo_zhu', 'hu_dong3', 'shi_jian_duan3'
])
print("\n2024年7月23日的预测结果：")
print(jie_guo_shu_ju_kuang)
jie_guo_shu_ju_kuang.to_csv('yu_ce_jie_guo_20240723.csv', index=False)

# 7. 生成时序图
for yong_hu in mu_biao_yong_hu:
    yong_hu_shu_ju = data[data['yong_hu_ID'] == yong_hu].sort_values('shi_jian')
    plt.figure(figsize=(12, 6))
    zhen_shi_shi_jian = yong_hu_shu_ju['shi_jian']
    zhen_shi_xing_wei = yong_hu_shu_ju['yong_hu_xing_wei']
    scatter = plt.scatter(zhen_shi_shi_jian, zhen_shi_xing_wei, c=zhen_shi_xing_wei, cmap='tab10', label=[f'行为{i}' for i in range(1, 5)], s=50)
    if yong_hu in jie_guo_shu_ju_kuang['yong_hu_ID'].values and jie_guo_shu_ju_kuang.loc[jie_guo_shu_ju_kuang['yong_hu_ID'] == yong_hu, 'shi_fou_zai_xian (1=是, 0=否)'].iloc[0] == 1:
        yu_ce_shi_jian = pd.date_range(start='2024-07-23 00:00:00', end='2024-07-23 23:00:00', freq='1H')
        yong_hu_shi_jian_duan = yong_hu_shu_ju.groupby('shi_jian_duan')['yong_hu_xing_wei'].value_counts(normalize=True).unstack(fill_value=0)
        hu_dong_values = []
        for shi_jian in yu_ce_shi_jian:
            duan = fen_pei_shi_jian_duan(shi_jian.hour)
            duan_xu_lie = xu_lie.copy()
            for seg in shi_jian_duan:
                duan_xu_lie[:, :, te_zheng.index(f'shi_jian_duan_{seg}')] = 1 if seg == duan else 0
            duan_xu_lie[:, :, te_zheng.index('xiao_shi')] = shi_jian.hour
            duan_xu_lie[:, :, te_zheng.index('ri_qi')] = 23
            duan_xu_lie_zhang_liang = torch.tensor(duan_xu_lie, dtype=torch.float32).to(she_bei)
            with torch.no_grad():
                _, hu_dong_yu_ce = mo_xing(duan_xu_lie_zhang_liang)
            hu_dong = max(0, hu_dong_yu_ce.cpu().numpy().squeeze().item())
            hu_dong_values.append(hu_dong)
        hu_dong_values = pd.Series(hu_dong_values).ewm(span=3).mean().values
        hu_dong_values = np.array(hu_dong_values)
        thresholds = np.percentile(hu_dong_values, [25, 50, 75])
        t_low, t_mid, t_high = thresholds
        yu_ce_xing_wei = []
        for i, shi_jian in enumerate(yu_ce_shi_jian):
            duan = fen_pei_shi_jian_duan(shi_jian.hour)
            hu_dong = hu_dong_values[i]
            if duan in yong_hu_shi_jian_duan.index:
                xing_wei_gai_lv = yong_hu_shi_jian_duan.loc[duan].to_dict()
            else:
                xing_wei_gai_lv = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
            if hu_dong >= t_high:
                xing_wei_gai_lv[4] *= 1.5
                xing_wei_gai_lv[3] *= 1.2
            elif hu_dong >= t_mid:
                xing_wei_gai_lv[3] *= 1.5
                xing_wei_gai_lv[2] *= 1.2
            elif hu_dong >= t_low:
                xing_wei_gai_lv[2] *= 1.5
                xing_wei_gai_lv[1] *= 1.2
            else:
                xing_wei_gai_lv[1] *= 1.5
            gai_lv_sum = sum(xing_wei_gai_lv.values())
            xing_wei_gai_lv = {k: v/gai_lv_sum for k, v in xing_wei_gai_lv.items()}
            xing_wei = np.random.choice([1, 2, 3, 4], p=[xing_wei_gai_lv.get(i, 0) for i in [1, 2, 3, 4]])
            yu_ce_xing_wei.append(xing_wei)
        plt.scatter(yu_ce_shi_jian, yu_ce_xing_wei, marker='^', c='red', label='预测行为', s=100)
    plt.axvline(x=pd.to_datetime('2024-07-20'), color='r', linestyle='--', label='预测点')
    plt.title(f'用户 {yong_hu} 的行为时间线')
    plt.xlabel('时间')
    plt.ylabel('行为类型')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'time_line_{yong_hu}.png')
    plt.close()
print("时序图已保存为 time_line_U*.png")
print("预测结果已保存至 yu_ce_jie_guo_20240723.csv")