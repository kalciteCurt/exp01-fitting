import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

def true_line(x):
    return 2 * x + 1

n_inliers, n_outliers = 50, 20

x_in = np.linspace(0, 10, n_inliers)
y_in = true_line(x_in) + np.random.normal(0, 0.5, n_inliers)

x_out = np.random.uniform(0, 10, n_outliers)
y_out = np.random.uniform(-5, 25, n_outliers)

x_all = np.concatenate([x_in, x_out])
y_all = np.concatenate([y_in, y_out])

plt.scatter(x_in, y_in, c='g', label='内点')
plt.scatter(x_out, y_out, c='r', marker='x', label='外点')
plt.legend()
plt.title('直线数据：内点 + 外点')
plt.show()


A = np.c_[x_all, np.ones_like(x_all)]     # 设计矩阵 [x, 1]
w_l2 = np.linalg.lstsq(A, y_all, rcond=None)[0]
k_l2, b_l2 = w_l2

print(f'L2 最小二乘：k = {k_l2:.3f}, b = {b_l2:.3f}')


def l1_loss(params):
    k, b = params
    return np.sum(np.abs(y_all - (k * x_all + b)))

res = minimize(l1_loss, x0=[0.0, 0.0], method='Nelder-Mead')
k_l1, b_l1 = res.x

print(f'L1 最小化：k = {k_l1:.3f}, b = {b_l1:.3f}')


xx = np.linspace(0, 10, 100)

plt.scatter(x_in, y_in, c='g', label='内点')
plt.scatter(x_out, y_out, c='r', marker='x', label='外点')
plt.plot(xx, k_l2 * xx + b_l2, 'b-', label=f'L2: k={k_l2:.2f}, b={b_l2:.2f}')
plt.plot(xx, k_l1 * xx + b_l1, 'm-', label=f'L1: k={k_l1:.2f}, b={b_l1:.2f}')
plt.plot(xx, true_line(xx), 'k--', label='真实 y = 2x + 1')
plt.legend()
plt.title('L2 vs L1：外点下 L1 更稳')
plt.show()

# 量化对比：k、b 与真值的误差
print('真实 k=2.0, b=1.0')
print(f'L2 误差: |Δk|={abs(k_l2-2.0):.3f}, |Δb|={abs(b_l2-1.0):.3f}')
print(f'L1 误差: |Δk|={abs(k_l1-2.0):.3f}, |Δb|={abs(b_l1-1.0):.3f}')



X = x_all.reshape(-1, 1)
y = y_all

ransac = RANSACRegressor(
    estimator=LinearRegression(),
    residual_threshold=1.0,
    random_state=0
)
ransac.fit(X, y)

k_r = ransac.estimator_.coef_[0]
b_r = ransac.estimator_.intercept_
inlier_mask = ransac.inlier_mask_

print(f'RANSAC：k = {k_r:.3f}, b = {b_r:.3f}')
print(f'RANSAC 找到内点数：{inlier_mask.sum()} / {len(y)}')

plt.scatter(x_all[inlier_mask], y_all[inlier_mask], c='g', label='RANSAC 判定内点')
plt.scatter(x_all[~inlier_mask], y_all[~inlier_mask], c='r', marker='x', label='外点')
plt.plot(xx, k_r * xx + b_r, 'b-', label=f'RANSAC: k={k_r:.2f}, b={b_r:.2f}')
plt.plot(xx, true_line(xx), 'k--', label='真实 y = 2x + 1')
plt.legend()
plt.title('RANSAC 去外点后拟合')
plt.show()



def true_func(x):
    return np.sin(2 * np.pi * x)

n_train = 50
x_train = np.linspace(0, 1, n_train).astype(np.float32)
y_train = (true_func(x_train) + np.random.normal(0, 0.2, n_train)).astype(np.float32)

x_plot = np.linspace(0, 1, 300).astype(np.float32)

Xtr = torch.tensor(x_train).view(-1, 1)
Ytr = torch.tensor(y_train).view(-1, 1)
Xpl = torch.tensor(x_plot).view(-1, 1)



class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

mlp = MLP()
optimizer = optim.Adam(mlp.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

losses = []
for epoch in range(3000):
    pred = mlp(Xtr)
    loss = loss_fn(pred, Ytr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f'最终训练 loss = {losses[-1]:.5f}')


y_mlp = mlp(Xpl).detach().numpy().ravel()

plt.plot(x_plot, true_func(x_plot), 'g--', label='真实 sin')
plt.plot(x_plot, y_mlp, 'r-', label='MLP 拟合')
plt.scatter(x_train, y_train, c='b', label='训练点')
plt.legend()
plt.title('PyTorch MLP 拟合 sin 函数')
plt.show()


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

x_train_2d = x_train.reshape(-1, 1)

# 多项式 9 次 + Ridge 正则
poly_model = make_pipeline(PolynomialFeatures(degree=9), Ridge(alpha=1e-3))
poly_model.fit(x_train_2d, y_train)
y_poly = poly_model.predict(x_plot.reshape(-1, 1))

plt.plot(x_plot, true_func(x_plot), 'g--', label='真实 sin')
plt.scatter(x_train, y_train, c='b', label='训练点')
plt.plot(x_plot, y_mlp, 'r-', label='MLP')
plt.plot(x_plot, y_poly, 'm-', label='多项式(9次)+Ridge')
plt.legend()
plt.title('MLP vs 多项式拟合')
plt.show()

# 训练误差对比
mlp_train_rmse = np.sqrt(mean_squared_error(y_train, mlp(Xtr).detach().numpy().ravel()))
poly_train_rmse = np.sqrt(mean_squared_error(y_train, poly_model.predict(x_train_2d)))
print(f'MLP  训练 RMSE = {mlp_train_rmse:.4f}')
print(f'多项式 训练 RMSE = {poly_train_rmse:.4f}')


