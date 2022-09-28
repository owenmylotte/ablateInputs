# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:32:01 2022

@author: Owen
"""
import numpy as np
import matplotlib.pyplot as plt  # for plotting


def l2norm(numerical, analytical):
    l2 = 0
    for i in range(numerical.size):
        l2 += ((numerical[i] - analytical[i]) / analytical[i]) ** 2
    l2 = np.sqrt(l2)
    return l2


data5 = np.loadtxt("anothertest5.txt", delimiter=' ', skiprows=1, dtype=float)
data10 = np.loadtxt("anothertest10.txt", delimiter=' ', skiprows=1, dtype=float)
data25 = np.loadtxt("anothertest25.txt", delimiter=' ', skiprows=1, dtype=float)
data50 = np.loadtxt("anothertest50.txt", delimiter=' ', skiprows=1, dtype=float)
data100 = np.loadtxt("anothertest100.txt", delimiter=' ', skiprows=1, dtype=float)
data250 = np.loadtxt("anothertest250.txt", delimiter=' ', skiprows=1, dtype=float)
data500 = np.loadtxt("anothertest500.txt", delimiter=' ', skiprows=1, dtype=float)
data1000 = np.loadtxt("anothertest1000.txt", delimiter=' ', skiprows=1, dtype=float)
data2000 = np.loadtxt("anothertest2000.txt", delimiter=' ', skiprows=1, dtype=float)
data5000 = np.loadtxt("anothertest5000.txt", delimiter=' ', skiprows=1, dtype=float)

height = np.zeros(1, dtype=float)
G = np.zeros(1, dtype=float)

## Results stuff
plt.figure(figsize=(6, 4), num=1)
plt.title("Parallel Plates Irradiation", pad=1)  # TITLE HERE

# plt.scatter(data5[:, 0], data5[:, 3], c='blue', s=4)
plt.scatter(data50[:, 0], data50[:, 3], c='blue', s=4)
plt.scatter(data100[:, 0], data100[:, 3], c='green', s=4)
plt.scatter(data250[:, 0], data250[:, 3], c='yellow', s=4)
plt.scatter(data500[:, 0], data500[:, 3], c='orange', s=4)
plt.scatter(data1000[:, 0], data1000[:, 3], c='red', s=4)
plt.scatter(data2000[:, 0], data2000[:, 3], c='purple', s=4)
# plt.scatter(data5000[:, 0], data5000[:, 3], c='black', s=4)
plt.plot(data500[:, 0], data500[:, 5], c='black', linewidth=2)

plt.legend(["dTheta = 3.6", "dTheta = 1.8", "dTheta = 0.72", "dTheta = 0.36", "dTheta = 0.18", "dTheta = 0.09", "Analytical Solution"],
           loc="upper right", prop={'size': 7})
plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('Postion [meters]', fontsize=10)
plt.ylabel('Irradiation [Watts]', fontsize=10)
plt.savefig('PlatesValidation1D', dpi=1000, bbox_inches='tight')
plt.show()

## Error stuff
ref_irradiation = [5 * 10, 10 * 20, 25 * 50, 50 * 100, 100 * 200, 250 * 500, 500 * 1000, 1000 * 2000, 2000 * 4000,
                   5000 * 10000]
l2_irradiation = np.array(
    [l2norm(data5[:, 3], data100[:, 5]), l2norm(data10[:, 3], data100[:, 5]), l2norm(data25[:, 3], data100[:, 5]),
     l2norm(data50[:, 3], data100[:, 5]), l2norm(data100[:, 3], data100[:, 5]), l2norm(data250[:, 3], data250[:, 5]),
     l2norm(data500[:, 3], data500[:, 5]), l2norm(data1000[:, 3], data500[:, 5]), l2norm(data2000[:, 3], data2000[:, 5]),
     l2norm(data5000[:, 3], data5000[:, 5])])

plt.figure(figsize=(6, 4), num=2)
plt.title("Parallel Plates Irradiation Convergence", pad=1)  # TITLE HERE
plt.loglog(ref_irradiation, l2_irradiation, c='blue', linewidth=1, marker='.')
plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('Ray Number', fontsize=10)
plt.ylabel('Error', fontsize=10)
plt.savefig('PlatesIrradiation_Convergence', dpi=1000, bbox_inches='tight')
plt.show()

# %% Radiative Equilibrium Plot
dataSimit = np.loadtxt("PlanarEquilibrium01.txt", delimiter=' ', skiprows=0, dtype=float)
dataAblate_500rays = np.loadtxt("equilibriumResult01_500rays.txt", delimiter=' ', skiprows=1, dtype=float)
dataAblate_250rays = np.loadtxt("equilibriumResult01_250rays.txt", delimiter=' ', skiprows=1, dtype=float)
dataAblate_100rays = np.loadtxt("equilibriumResult01_100rays.txt", delimiter=' ', skiprows=1, dtype=float)
dataAblate_50rays = np.loadtxt("equilibriumResult01_50rays.txt", delimiter=' ', skiprows=1, dtype=float)
dataAblate_25rays = np.loadtxt("equilibriumResult01_25rays.txt", delimiter=' ', skiprows=1, dtype=float)
dataAblate_15rays = np.loadtxt("equilibriumResult01_15rays.txt", delimiter=' ', skiprows=1, dtype=float)
dataAblate_10rays = np.loadtxt("equilibriumResult01_10rays.txt", delimiter=' ', skiprows=1, dtype=float)
dataAblate_5rays = np.loadtxt("equilibriumResult01_5rays.txt", delimiter=' ', skiprows=1, dtype=float)
dataAblate10n = np.loadtxt("equilibriumResult10.txt", delimiter=' ', skiprows=1, dtype=float)

plt.figure(figsize=(6, 4), num=3)
plt.title("Radiative Equilibrium", pad=1)  # TITLE HERE
plt.plot(dataSimit[:, 0], dataSimit[:, 1], c='black', linewidth=1.5)
plt.scatter(dataAblate_500rays[:, 0], dataAblate_500rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_250rays[:, 0], dataAblate_250rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_100rays[:, 0], dataAblate_100rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_50rays[:, 0], dataAblate_50rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_25rays[:, 0], dataAblate_25rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_15rays[:, 0], dataAblate_15rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_10rays[:, 0], dataAblate_10rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_5rays[:, 0], dataAblate_5rays[:, 5], c='red', s=4)
plt.scatter(dataAblate10n[:, 0], dataAblate10n[:, 5], c='green', s=4)

plt.legend(["Simit", "ABLATE (n = 1)", "ABLATE (n = 10)"], loc="upper right", prop={'size': 7})
plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('Postion [meters]', fontsize=10)
plt.ylabel('Temperature [K]', fontsize=10)
plt.ylim(1400, 1900)
plt.savefig('EquilibriumValidation1D_Parallel', dpi=1000, bbox_inches='tight')
plt.show()

## Error Stuff
ref_equilibrium = [5 * 10, 10 * 20, 15 * 30, 25 * 50, 50 * 100, 100 * 200, 250 * 500, 500 * 1000]
l2_equilibrium = np.array(
    [l2norm(dataAblate_5rays[:, 5], dataSimit[:, 1]), l2norm(dataAblate_10rays[:, 5], dataSimit[:, 1]),
     l2norm(dataAblate_15rays[:, 5], dataSimit[:, 1]), l2norm(dataAblate_25rays[:, 5], dataSimit[:, 1]),
     l2norm(dataAblate_50rays[:, 5], dataSimit[:, 1]),
     l2norm(dataAblate_100rays[:, 5], dataSimit[:, 1]), l2norm(dataAblate_250rays[:, 5], dataSimit[:, 1]),
     l2norm(dataAblate_500rays[:, 5], dataSimit[:, 1])])

plt.figure(figsize=(6, 4), num=4)
plt.title("Radiative Equilibrium Convergence", pad=1)  # TITLE HERE
plt.loglog(ref_equilibrium, l2_equilibrium, c='blue', linewidth=1, marker='.')
plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('Ray Number', fontsize=10)
plt.ylabel('Error', fontsize=10)
plt.savefig('Equilibrium_Convergence', dpi=1000, bbox_inches='tight')
plt.show()

# %% Diffusion Flame Verification
diffusionFlame = np.loadtxt("diffusionFlame01.txt", delimiter=' ', skiprows=1,
                            dtype=float)  # Import data, skip first row (header), float data format
flameSimit = np.loadtxt("diffusionFlameSimit.txt", delimiter=' ', skiprows=1,
                        dtype=float)  # Import data, skip first row (header), float data format

plt.figure(figsize=(6, 4), num=5)
plt.title("Diffusion Flame with Radiation", pad=1)  # TITLE HERE

plt.plot(flameSimit[:, 0], flameSimit[:, 1], c='black', linewidth=1)
plt.scatter(diffusionFlame[:, 0], diffusionFlame[:, 5], c='red', s=4)

plt.legend(["Simit", "ABLATE"],
           loc="upper left", prop={'size': 7})
plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('Postion [meters]', fontsize=10)
plt.ylabel('Temperature [K]', fontsize=10)
plt.savefig('DiffusionFlameValidation', dpi=1000, bbox_inches='tight')
plt.show()

# # %% Error Convergence Plot

print(l2norm(diffusionFlame[:, 5], flameSimit[:, 1]))
