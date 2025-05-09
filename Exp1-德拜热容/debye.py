import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

# 物理常数
k_B = 1.380649e-23  # 玻尔兹曼常数，单位：J/K

# 铝的参数
V = 1000 * 1e-6  # 体积，从cm³转换为m³
rho = 6.022e28  # 原子数密度，单位：m⁻³
theta_D = 428  # 德拜温度，单位：K

def cv_with_details(T):
    """
    计算给定温度下的热容，并返回积分上限、积分值和热容
    :param T: 温度 (K)
    :return: (积分上限, 积分值, 热容)
    """
    # 高斯积分点和权重 (50个点)
    points, weights = leggauss(50)
    
    # 积分上限
    upper_limit = theta_D / T
    
    # 变换积分区间从[-1,1]到[0,upper_limit]
    x = 0.5 * upper_limit * (points + 1)
    dx = 0.5 * upper_limit * weights
    
    # 计算被积函数，注意处理数值稳定性
    exp_x = np.exp(x)
    # 对于x较大的情况，使用近似处理避免数值溢出
    mask = x > 100  # 当x很大时，e^x/(e^x-1)^2 ≈ e^-x
    integrand = np.zeros_like(x)
    integrand[~mask] = (x[~mask]**4 * exp_x[~mask]) / (exp_x[~mask] - 1)**2
    integrand[mask] = x[mask]**4 * np.exp(-x[mask])  # 近似处理
    
    integral = np.sum(integrand * dx)
    
    # 计算热容
    C_V = 9 * V * rho * k_B * (T / theta_D)**3 * integral
    
    return upper_limit, integral, C_V

# 指定温度点
specified_temperatures = [5, 50, 100, 300, 500, 1000]  # 单位：K

# 计算并显示指定温度点的结果
print("温度(K) | 积分上限 | 积分值 | 热容(J/K)")
print("----------------------------------------")
for T in specified_temperatures:
    upper_limit, integral, C_V = cv_with_details(T)
    print(f"{T:6.1f} | {upper_limit:8.3f} | {integral:8.5f} | {C_V:.3e}")

# 计算温度范围内的热容用于绘图
temperatures = np.linspace(5, 500, 100)  # 从5K到500K
heat_capacities = np.array([cv_with_details(T)[2] for T in temperatures])

# 绘制热容曲线
plt.figure(figsize=(10, 6))
plt.plot(temperatures, heat_capacities, 'b-', linewidth=2)
plt.xlabel('Temperature (K)', fontsize=14)
plt.ylabel('Heat Capacity $C_V$ (J/K)', fontsize=14)
plt.title('Heat Capacity of Aluminum According to Debye Model', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(axis='both', which='major', labelsize=12)

# 标记德拜温度
plt.axvline(x=theta_D, color='r', linestyle='--', label=f'Debye Temperature ({theta_D} K)')
plt.legend(fontsize=12)

plt.show()
