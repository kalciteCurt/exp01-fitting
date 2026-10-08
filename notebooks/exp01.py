import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  
plt.rcParams['axes.unicode_minus'] = False     

torch.manual_seed(12)
np.random.seed(12)
plt.rcParams['figure.figsize'] = (8,5)

n_inliers = 50
n_outliers = 20
x_in = np.linspace(0,10,n_inliers)
y_in = 2*x_in + 1 + np.random.normal(0,0.5,n_inliers)

x_out = np.random.uniform(0,10,n_outliers)
y_out = np.random.uniform(-5,25,n_outliers)

x_all = np.concatenate([x_in,x_out])
y_all = np.concatenate([y_in,y_out])

model = LinearRegression()
model.fit(x)
y_pred = model.predict(x_all)


plt.scatter(x_in, y_in, c='g', label='inliers 内点')
plt.scatter(x_out, y_out, c='r', marker='x', label='outliers 外点')
plt.legend()
plt.title('直线数据：内点 + 外点')
plt.show()


