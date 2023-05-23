# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:32:01 2022

@author: Owen
"""
import numpy as np
import matplotlib.pyplot as plt  # for plotting
import matplotlib.font_manager
import matplotlib.ticker as ticker

# plt.style.use('ggplot')
plt.rcParams["font.family"] = "Noto Serif CJK JP"


def l2norm(numerical, analytical):
    l2 = 0
    for i in range(numerical.size):
        l2 += ((numerical[i] - analytical[i]) / analytical[i]) ** 2
    l2 = np.sqrt(l2)
    return l2


savePath = "/home/owen/CLionProjects/ParallelRadiationJCP/figures/"

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
# plt.title("Parallel Plates Irradiation", pad=1, fontsize=10)  # TITLE HERE

# plt.scatter(data5[:, 0], data5[:, 3], c='blue', s=4)
plt.plot(data50[:, 0], data50[:, 3], c='black', linestyle="dotted", linewidth=1)  # , c='blue', s=4)
plt.plot(data100[:, 0], data100[:, 3], c='black', linestyle="dashdot", linewidth=1)  # , c='green', s=4)
# plt.scatter(data250[:, 0], data250[:, 3], s=4)  # , c='yellow', s=4)
# plt.scatter(data500[:, 0], data500[:, 3], s=4)  # , c='orange', s=4)
# plt.scatter(data2000[:, 0], data2000[:, 3], c='purple', s=4)
plt.plot(data1000[:, 0], data1000[:, 3], c='black', linestyle="dashed", linewidth=1)  # , c='red', s=4)
# plt.scatter(data5000[:, 0], data5000[:, 3], c='black', s=4)
plt.plot(data5000[:, 0], data5000[:, 5], c='black', linewidth=1)

plt.legend([r"$d \theta = 3.6$", r"$d \theta = 1.8$", r"$d \theta = 0.18$",
            "Analytical Solution"],
           # , r"$d \theta = 0.09$", "Analytical Solution"],
           loc="upper right", prop={'size': 7}, frameon=False)

plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('x $[m]$', fontsize=10)
plt.ylabel(r'I $[\frac{W}{m^3}]$', fontsize=10)
plt.ticklabel_format(style="sci")
plt.savefig(savePath + 'PlatesValidation1D', dpi=1000, bbox_inches='tight')
plt.show()

## Results for comparison mesh refinement
irrMesh10 = np.loadtxt("irrMesh_10.txt", delimiter=' ', skiprows=1, dtype=float)
irrMesh50 = np.loadtxt("irrMesh_50.txt", delimiter=' ', skiprows=1, dtype=float)
irrMesh100 = np.loadtxt("irrMesh_100.txt", delimiter=' ', skiprows=1, dtype=float)
plt.plot(irrMesh10[:, 0], irrMesh10[:, 3], c='black', linestyle="dotted", linewidth=1)  # , c='blue', s=4)
plt.plot(irrMesh10[:, 0], irrMesh10[:, 3], c='black', linestyle="dashdot", linewidth=1)  # , c='green', s=4)
plt.plot(irrMesh10[:, 0], irrMesh10[:, 3], c='black', linestyle="dashed", linewidth=1)  # , c='red', s=4)
plt.plot(data5000[:, 0], data5000[:, 5], c='black', linewidth=1)
plt.legend([r"$dx = 10^{-3}$", r"$dx = 5 \times 10^{-4}$", r"$dx = 10^{-4}$",
            "Analytical Solution"], loc="upper right", prop={'size': 7}, frameon=False)

plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('x $[m]$', fontsize=10)
plt.ylabel(r'I $[\frac{W}{m^3}]$', fontsize=10)
plt.ticklabel_format(style="sci")
plt.savefig(savePath + 'PlatesValidation1D_Mesh', dpi=1000, bbox_inches='tight')
plt.show()

## Ray Error stuff
ref_irradiation = [5 * 10, 10 * 20, 25 * 50, 50 * 100, 100 * 200, 250 * 500, 500 * 1000, 1000 * 2000, 2000 * 4000,
                   5000 * 10000]
l2_irradiation = np.array(
    [l2norm(data5[:, 3], data100[:, 5]), l2norm(data10[:, 3], data100[:, 5]), l2norm(data25[:, 3], data100[:, 5]),
     l2norm(data50[:, 3], data100[:, 5]), l2norm(data100[:, 3], data100[:, 5]), l2norm(data250[:, 3], data250[:, 5]),
     l2norm(data500[:, 3], data500[:, 5]), l2norm(data1000[:, 3], data2000[:, 5]),
     l2norm(data2000[:, 3], data2000[:, 5]),
     l2norm(data5000[:, 3], data5000[:, 5])])

plt.figure(figsize=(6, 4), num=2)
# plt.title("Irradiation Ray Convergence", pad=1, fontsize=10)  # TITLE HERE
plt.loglog(ref_irradiation[0:8], l2_irradiation[0:8], c='black', linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('N', fontsize=10)
plt.ylabel(r'$\epsilon_I$', fontsize=10)
plt.savefig(savePath + 'PlatesIrradiation_Convergence', dpi=1000, bbox_inches='tight')
plt.show()

## Mesh Convergence Stuff
irrMeshes = [10, 50, 100]
l2irrMeshes = np.zeros(len(irrMeshes))
for i in range(len(irrMeshes)):
    analyticalSolution = np.zeros(irrMeshes[i])
    for n in range(irrMeshes[i]):
        skip = 200 / irrMeshes[i]
        analyticalSolution[n] = data5000[int(skip * n), 5]
    irrMeshData = np.loadtxt('irrMesh_' + str(irrMeshes[i]) + ".txt", delimiter=' ', skiprows=1, dtype=float)
    l2irrMeshes[i] = l2norm(irrMeshData[:, 3], analyticalSolution)
print(l2irrMeshes)
plt.figure(figsize=(6, 4), num=2)
# plt.title("Irradiation Mesh Convergence", pad=1, fontsize=10)  # TITLE HERE
plt.loglog(irrMeshes, l2irrMeshes, c='black', linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
# Define your custom formatter function


def minor_formatter(x, pos):
    return f'{int(x)}'


# Set minor formatter for both axes
plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(minor_formatter))
plt.gca().tick_params(axis='x', which='minor', labelsize=8.5)  # set fontsize for minor ticks
plt.xlabel('N', fontsize=10)
plt.ylabel(r'$\epsilon_I$', fontsize=10)
plt.savefig(savePath + 'PlatesIrradiation_MeshConvergence', dpi=1000, bbox_inches='tight')
plt.show()

## Scaling Stuff
t_solve = [6.26E+01 / 6.26E+01, 6.26E+01 / 4.10E+01, 6.26E+01 / 3.48E+01, float("nan"), float("nan"), float("nan"),
           float("nan"), 6.26E+01 / 2.41E+01, 6.26E+01 / 3.34E+01,
           float("nan"), float("nan")]
t_init = [1.76E+03 / 1.76E+03, 1.76E+03 / 9.98E+02, 1.76E+03 / 6.74E+02, float("nan"), float("nan"), float("nan"),
          float("nan"), 1.76E+03 / 1.96E+02, 1.76E+03 / 2.86E+02,
          float("nan"), float("nan")]
n_proc = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

plt.figure(figsize=(6, 4), num=3)
# plt.title("Parallel Plates Irradiation Scaling", pad=1, fontsize=10)  # TITLE HERE
plt.loglog(n_proc, t_init, c='blue', linewidth=1, marker='.')
plt.loglog(n_proc, t_solve, c='orange', linewidth=1, marker='.')
plt.loglog(n_proc, n_proc, c='black', linewidth=1)
plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('# Processes', fontsize=10)
plt.ylabel('Speedup', fontsize=10)
plt.legend(["Initialization (no sharing)", "Solve (no sharing)", "Ideal"], loc="upper left", prop={'size': 7}, frameon=False)

plt.savefig(savePath + 'PlatesIrradiation_Scaling', dpi=1000, bbox_inches='tight')
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
dataAblate10sh = np.loadtxt("equilibriumResult_raySharing10.txt", delimiter=' ', skiprows=1, dtype=float)

plt.figure(figsize=(6, 4), num=3)
# plt.title("Planar Radiative Equilibrium", pad=1, fontsize=10)  # TITLE HERE
plt.plot(dataAblate_5rays[:, 0], dataAblate_5rays[:, 5], c='black', linestyle="dotted", linewidth=1)
plt.plot(dataAblate_10rays[:, 0], dataAblate_10rays[:, 5], c='black', linestyle="dashdot", linewidth=1)
plt.plot(dataAblate_500rays[:, 0], dataAblate_500rays[:, 5], c='black', linestyle="dashed", linewidth=1)
# plt.scatter(dataAblate_250rays[:, 0], dataAblate_250rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_100rays[:, 0], dataAblate_100rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_50rays[:, 0], dataAblate_50rays[:, 5], s=4)
# plt.scatter(dataAblate_25rays[:, 0], dataAblate_25rays[:, 5], c='red', s=4)
# plt.scatter(dataAblate_15rays[:, 0], dataAblate_15rays[:, 5], s=4)
# plt.plot(dataAblate10n[:, 0], dataAblate10n[:, 5], c='black', linestyle="dashed", linewidth=1)
plt.plot(dataSimit[:, 0], dataSimit[:, 1], c='black', linewidth=1)
# plt.scatter(dataAblate10sh[:, 0], dataAblate10sh[:, 6], c='purple', s=4)

plt.legend([r"$d \theta = 36$", r"$d \theta = 18$", r"$d \theta = 0.36$", "Simit", ], loc="upper right",
           prop={'size': 7}, frameon=False)

plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('x $[m]$', fontsize=10)
plt.ylabel('T $[K]$', fontsize=10)
plt.ylim(1450, 1900)
plt.savefig(savePath + 'EquilibriumValidation1D', dpi=1000, bbox_inches='tight')
plt.show()

## Equilibrium Mesh Comparison Plot
eqMesh10 = np.loadtxt("radEq_10.txt", delimiter=' ', skiprows=1, dtype=float)
eqMesh50 = np.loadtxt("radEq_50.txt", delimiter=' ', skiprows=1, dtype=float)
eqMesh100 = np.loadtxt("radEq_100.txt", delimiter=' ', skiprows=1, dtype=float)
plt.figure(figsize=(6, 4), num=3)
plt.plot(eqMesh10[:, 0], eqMesh10[:, 5], c='black', linestyle="dotted", linewidth=1)
plt.plot(eqMesh50[:, 0], eqMesh50[:, 5], c='black', linestyle="dashdot", linewidth=1)
plt.plot(eqMesh100[:, 0], eqMesh100[:, 5], c='black', linestyle="dashed", linewidth=1)
plt.plot(dataSimit[:, 0], dataSimit[:, 1], c='black', linewidth=1)
plt.legend([r"$dx = 10^{-1}$", r"$dx = 5 \times 10^{-2}$", r"$dx = 10^{-2}$",
            "Simit"], loc="upper right",
           prop={'size': 7}, frameon=False)

plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('x $[m]$', fontsize=10)
plt.ylabel('T $[K]$', fontsize=10)
plt.ylim(1450, 1900)
plt.savefig(savePath + 'EquilibriumValidation1DMesh', dpi=1000, bbox_inches='tight')
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
# plt.title("Radiative Equilibrium Ray Convergence", pad=1, fontsize=10)  # TITLE HERE
plt.loglog(ref_equilibrium, l2_equilibrium, c='black', linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('N', fontsize=10)
plt.ylabel('$\epsilon_T$', fontsize=10)
plt.savefig(savePath + 'Equilibrium_Convergence', dpi=1000, bbox_inches='tight')
plt.show()

## Mesh Convergence Stuff
irrMeshes = [10, 50, 100, 200]
l2irrMeshes = np.zeros(len(irrMeshes))
for i in range(len(irrMeshes)):
    analyticalSolution = np.zeros(irrMeshes[i])
    for n in range(irrMeshes[i]):
        skip = 200 / irrMeshes[i]
        analyticalSolution[n] = dataSimit[int(skip * n), 1]
    irrMeshData = np.loadtxt('radEq_' + str(irrMeshes[i]) + ".txt", delimiter=' ', skiprows=1, dtype=float)
    l2irrMeshes[i] = l2norm(irrMeshData[:, 5], analyticalSolution)

plt.figure(figsize=(6, 4), num=4)
# plt.title("Radiative Equilibrium Mesh Convergence", pad=1, fontsize=10)  # TITLE HERE
plt.loglog(irrMeshes, l2irrMeshes, c='black', linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
# Define your custom formatter function


def minor_formatter(x, pos):
    return f'{int(x)}'


# Set minor formatter for both axes
plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(minor_formatter))
plt.gca().tick_params(axis='x', which='minor', labelsize=8.5)  # set fontsize for minor ticks
plt.ylabel('$\epsilon_T$', fontsize=10)
plt.savefig(savePath + 'Equilibrium_MeshConvergence', dpi=1000, bbox_inches='tight')
plt.show()

# ## Scaling Stuff
# t_solve_eq = [6.17E+03 / 6.17E+03, 6.17E+03 / 4.48E+03, 6.17E+03 / 3.37E+03, float("nan"), float("nan"), float("nan"),
#               float("nan"), 6.17E+03 / 2.36E+03, float("nan"),
#               float("nan"), float("nan")]
# t_init_eq = [1.76E+03 / 1.76E+03, 1.76E+03 / 9.93E+02, 1.76E+03 / 6.90E+02, float("nan"), float("nan"), float("nan"),
#              float("nan"), 1.76E+03 / 1.96E+02, float("nan"),
#              float("nan"), float("nan")]
# n_proc_eq = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#
# plt.figure(figsize=(6, 4), num=3)
# plt.title("Radiative Equilibrium Scaling", pad=1, fontsize=10)  # TITLE HERE
# plt.loglog(n_proc_eq, t_init_eq, c='blue', linewidth=1, marker='.')
# plt.loglog(n_proc_eq, t_solve_eq, c='orange', linewidth=1, marker='.')
# plt.loglog(n_proc_eq, n_proc_eq, c='black', linewidth=1)
# plt.yticks(fontsize=7)
# plt.xticks(fontsize=7)
# plt.xlabel('# Processes', fontsize=10)
# plt.ylabel('Speedup', fontsize=10)
# plt.legend(["Initialization (no sharing)", "Solve (no sharing)", "Ideal"], loc="upper left", prop={'size': 7})
# plt.savefig(savePath + 'Equilibrium_Scaling', dpi=1000, bbox_inches='tight')
# plt.show()

# %% Diffusion Flame Verification
# diffusionFlame = np.loadtxt("diffusionFlame01.txt", delimiter=' ', skiprows=1,
#                             dtype=float)  # Import data, skip first row (header), float data format
# flameSimit = np.loadtxt("diffusionFlameSimit.txt", delimiter=' ', skiprows=1,
#                         dtype=float)  # Import data, skip first row (header), float data format
#
# plt.figure(figsize=(6, 4), num=5)
# plt.title("1D Diffusion Flame with Radiation", pad=1, fontsize=10)  # TITLE HERE
#
# plt.plot(flameSimit[:, 0], flameSimit[:, 1], c='black', linewidth=1)
# plt.scatter(diffusionFlame[:, 0], diffusionFlame[:, 5], c='red', s=4)
#
# plt.legend(["Simit", "ABLATE"],
#            loc="upper left", prop={'size': 7})
# plt.yticks(fontsize=7)
# plt.xticks(fontsize=7)
# plt.xlabel('Postion $[m]$', fontsize=10)
# plt.ylabel('Temperature $[K]$', fontsize=10)
# plt.savefig(savePath + 'DiffusionFlameValidation', dpi=1000, bbox_inches='tight')
# plt.show()
#
# # # %% Error Convergence Plot
#
# print(l2norm(diffusionFlame[:, 5], flameSimit[:, 1]))
#
# ## Scaling Stuff

## Results stuff
plt.figure(figsize=(6, 4), num=1)
# plt.title("Parallel Plates Temperature", pad=1, fontsize=10)  # TITLE HERE

x = np.linspace(-0.01, 0.01, 1000)
temp = np.zeros(1000)
for i in range(len(x)):
    if x[i] < 0:
        temp[i] = (-(7E6 * x[i] * x[i]) + 2000.0)
    else:
        temp[i] = (-(13E6 * x[i] * x[i]) + 2000.0)

plt.plot(x, temp, c='black', linewidth=1)  # , c='blue', s=4)

plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('x $[m]$', fontsize=10)
plt.ylabel(r'T $[K]$', fontsize=10)
plt.ticklabel_format(style="sci")
plt.savefig(savePath + 'PlatesTemperature', dpi=1000, bbox_inches='tight')
plt.show()
