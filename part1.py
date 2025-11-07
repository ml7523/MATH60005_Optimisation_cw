import numpy as np
import matplotlib.pyplot as plt

# Read data
train = np.loadtxt('data/trainingIa.dat')
valid = np.loadtxt('data/validationIa.dat')

x_train, y_train = train[:,0], train[:,1]
x_valid, y_valid = valid[:,0], valid[:,1]

def fit(n, x_train, y_train, x_valid, y_valid):
    """
    :param n: order
    :param x_train: training data - x
    :param y_train: training data - y
    :param x_valid: validation data - x
    :param y_valid: validation data - y
    """
    A = np.vander(x_train, N=n, increasing=True)
    theta = np.linalg.inv(A.T @ A) @ A.T @ y_train
    V_theta = np.vander(x_valid, N=n, increasing=True) @ theta
    mse = np.mean((V_theta - y_valid)**2)
    return theta, mse

ns = np.arange(1, 21)
mses = []
thetas = {}
for n in ns:
    th, mse = fit(n, x_train, y_train, x_valid, y_valid)
    mses.append(mse)
    thetas[n] = th

# 找到最小满足 MSE <= 1e-3 的 n*
n_star = next((int(n) for n, e in zip(ns, mses) if e <= 1e-3), None)
print("n* =", n_star, "MSE =", mses[ns.tolist().index(n_star)] if n_star else None)

# 画 MSE–n
plt.figure(); plt.plot(ns, mses, marker='o'); plt.xlabel('degree n'); plt.ylabel('MSE (validation)'); plt.show()

# 在 n* 下做“训练点数–MSE”与函数曲线
if n_star is not None:
    m_all = len(xs_tr)
    sizes = np.linspace(max(10, n_star+2), m_all, num=10, dtype=int)
    mses_vs_m = []
    rng = np.random.default_rng(42)
    for mprime in sizes:
        idx = rng.choice(m_all, size=mprime, replace=False)
        th, mse = fit_and_mse(n_star, xs_tr[idx], y_tr[idx], xs_va, y_va)
        mses_vs_m.append(mse)
    plt.figure(); plt.plot(sizes, mses_vs_m, marker='o'); plt.xlabel('# training points'); plt.ylabel('MSE (validation)'); plt.show()

    # 画学到的函数
    th = thetas[n_star]
    grid = np.linspace(xs_tr.min(), xs_tr.max(), 400)
    yhat = vander(grid, n_star) @ th
    # 反缩放仅用于横坐标显示（如果你想用原始 x 轴，可以反变换）
    def inv_scale(z): return (z+1)*(xmax-xmin)/2 + xmin
    plt.figure()
    plt.scatter(inv_scale(xs_tr), y_tr, s=10, label='train')
    plt.scatter(inv_scale(xs_va), y_va, s=10, label='val')
    plt.plot(inv_scale(grid), yhat, label=f'fit (n={n_star})')
    plt.legend(); plt.xlabel('x'); plt.ylabel('V(x)'); plt.show()
