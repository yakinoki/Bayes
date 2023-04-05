import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

# データの生成
np.random.seed(123)
x = np.linspace(0, 1, 100)
y = 0.5 * x + np.random.normal(0, 0.1, size=100)

# モデルの定義
with pm.Model() as model:
    # 事前分布の設定
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=1)
    sigma = pm.HalfNormal('sigma', sd=1)
    
    # モデル式の設定
    mu = alpha + beta * x
    
    # 尤度の設定
    likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=y)
    
    # MCMCサンプリングによるパラメータ推定
    trace = pm.sample(1000, tune=1000)
    
# 結果の可視化
az.plot_trace(trace)
plt.show()
