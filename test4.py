import numpy as np
import matplotlib.pyplot as plt

# 定义 sigmoid 函数
def S(x, alpha, x0):
    return 1 / (1 + np.exp(-alpha * (x - x0)))

# 定义目标函数
def reward(phi_a_bar, d):
    return -10 * abs(phi_a_bar-0.2) * S(d, 1/50, 4000)

# 生成 φ 和 d 的值
phi_vals = np.linspace(0, 1, 200)       # φ ∈ [0, 1]
d_vals = [100, 200,  3500,4500,5000]       # 固定一些 d 值画多条线

plt.figure(figsize=(8, 5))

# 每条线表示一个固定 d
for d in d_vals:
    r_vals = reward(phi_vals, d)
    plt.plot(phi_vals, r_vals, label=f'd = {d} m')

plt.title("Reward vs φ (for various distances d)")
plt.xlabel("φ_a_bar (normalized angle)")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
