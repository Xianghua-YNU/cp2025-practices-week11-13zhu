import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# 计算电势函数（假设线电荷模型）
def electric_potential(x, y, z, lambda_=1.0, epsilon0=8.854e-12):
    def integrand(z_prime):
        r = np.sqrt(x**2 + y**2 + (z - z_prime)**2)
        return lambda_ / (4 * np.pi * epsilon0 * r)
    result, _ = quad(integrand, -10, 10)  # 积分区间可调整
    return result

# 计算电场（通过电势梯度）
def electric_field(x, y, z, lambda_=1.0, epsilon0=8.854e-12, h=1e-6):
    vx = (electric_potential(x+h, y, z, lambda_, epsilon0) - 
          electric_potential(x-h, y, z, lambda_, epsilon0)) / (2*h)
    vy = (electric_potential(x, y+h, z, lambda_, epsilon0) - 
          electric_potential(x, y-h, z, lambda_, epsilon0)) / (2*h)
    vz = (electric_potential(x, y, z+h, lambda_, epsilon0) - 
          electric_potential(x, y, z-h, lambda_, epsilon0)) / (2*h)
    return -np.array([vx, vy, vz])  # 取负得到电场

# 可视化（yz平面，x=0）
y = np.linspace(-5, 5, 50)
z = np.linspace(-5, 5, 50)
Y, Z = np.meshgrid(y, z)
X = np.zeros_like(Y)

# 计算电势分布
V = np.array([[electric_potential(0, yi, zi) for yi, zi in zip(row_y, row_z)] 
              for row_y, row_z in zip(Y, Z)])

# 绘制等势线
plt.figure(figsize=(10, 6))
plt.contour(Y, Z, V, levels=20, colors='black')

# 计算电场分布并绘制矢量
E = np.array([[electric_field(0, yi, zi)[:2] for yi, zi in zip(row_y, row_z)] 
              for row_y, row_z in zip(Y, Z)])
plt.quiver(Y, Z, E[:,:,0], E[:,:,1], scale=100)

plt.xlabel('y')
plt.ylabel('z')
plt.title('Electric Potential Contour and Electric Field Vectors')
plt.show()
