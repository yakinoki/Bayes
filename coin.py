import pymc3 as pm
import numpy as np

# コイン投げデータ
data = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1])

# モデルの定義
with pm.Model() as coin_model:
    # 事前分布
    p = pm.Beta('p', alpha=1, beta=1)
    
    # 尤度関数
    y = pm.Bernoulli('y', p=p, observed=data)
    
    # MCMCサンプリング
    trace = pm.sample(1000, chains=4)

# 結果のプロット
pm.plot_trace(trace)

