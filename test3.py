import numpy as np
import matplotlib.pyplot as plt

# 定义 Sigmoid 函数
def S(x, alpha, x0):
    return 1 / (1 + np.exp(-alpha * (x - x0)))

# 设置参数
alpha = 1/200 # 控制斜率
x0 = 3000     # 控制偏移

# 定义 x 的取值范围
x = np.linspace(0, 10000, 5000)

# 计算 y 值
y = S(x, alpha, x0)

# 绘图
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=f'Sigmoid: α={alpha}, x₀={x0}', color='blue')
plt.title('Sigmoid Function S(x, α, x₀)')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.grid(True)
plt.legend()
plt.axvline(x0, color='gray', linestyle='--', label='x₀')
plt.show()
