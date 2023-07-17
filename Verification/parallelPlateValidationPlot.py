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
plt.plot(data50[:, 0], data50[:, 3], c='black', linestyle="dotted", linewidth=1)  # , c='blue', s=4)
plt.plot(data100[:, 0], data100[:, 3], c='black', linestyle="dashdot", linewidth=1)  # , c='green', s=4)
plt.plot(data1000[:, 0], data1000[:, 3], c='black', linestyle="dashed", linewidth=1)  # , c='red', s=4)
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
# TODO: Change the bounding box to be equal width on both sides of the plot
plt.show()

# Ray Error stuff
ref_irradiation = [5 * 10, 10 * 20, 25 * 50, 50 * 100, 100 * 200, 250 * 500, 500 * 1000, 1000 * 2000, 2000 * 4000,
                   5000 * 10000]
l2_irradiation = np.array(
    [l2norm(data5[:, 3], data100[:, 5]), l2norm(data10[:, 3], data100[:, 5]), l2norm(data25[:, 3], data100[:, 5]),
     l2norm(data50[:, 3], data100[:, 5]), l2norm(data100[:, 3], data100[:, 5]), l2norm(data250[:, 3], data250[:, 5]),
     l2norm(data500[:, 3], data500[:, 5]), l2norm(data1000[:, 3], data2000[:, 5]),
     l2norm(data2000[:, 3], data2000[:, 5]),
     l2norm(data5000[:, 3], data5000[:, 5])])

plt.figure(figsize=(6, 4), num=2)
plt.loglog(ref_irradiation[0:8], l2_irradiation[0:8], c='black', linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('N', fontsize=10)
plt.ylabel(r'$\epsilon_I$', fontsize=10)
plt.savefig(savePath + 'PlatesIrradiation_Convergence', dpi=1000, bbox_inches='tight')
# TODO: Change the bounding box to be equal width on both sides of the plot
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


def minor_formatter(x, pos):
    return f'{int(x)}'


# Set minor formatter for both axes
plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(minor_formatter))
plt.gca().tick_params(axis='x', which='minor', labelsize=8.5)  # set fontsize for minor ticks
plt.xlabel('N', fontsize=10)
plt.ylabel(r'$\epsilon_I$', fontsize=10)
plt.savefig(savePath + 'PlatesIrradiation_MeshConvergence', dpi=1000, bbox_inches='tight')
# TODO: Change the bounding box to be equal width on both sides of the plot
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
plt.plot(dataAblate_5rays[:, 0], dataAblate_5rays[:, 5], c='black', linestyle="dotted", linewidth=1)
plt.plot(dataAblate_10rays[:, 0], dataAblate_10rays[:, 5], c='black', linestyle="dashdot", linewidth=1)
plt.plot(dataAblate_500rays[:, 0], dataAblate_500rays[:, 5], c='black', linestyle="dashed", linewidth=1)
plt.plot(dataSimit[:, 0], dataSimit[:, 1], c='black', linewidth=1)

plt.legend([r"$d \theta = 36$", r"$d \theta = 18$", r"$d \theta = 0.36$", "Reference Solution", ], loc="upper right",
           prop={'size': 7}, frameon=False)

plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('x $[m]$', fontsize=10)
plt.ylabel('T $[K]$', fontsize=10)
plt.ylim(1450, 1900)
plt.savefig(savePath + 'EquilibriumValidation1D', dpi=1000, bbox_inches='tight')
plt.show()

# Error Stuff
ref_equilibrium = [5 * 10, 10 * 20, 15 * 30, 25 * 50, 50 * 100, 100 * 200, 250 * 500, 500 * 1000]
l2_equilibrium = np.array(
    [l2norm(dataAblate_5rays[:, 5], dataSimit[:, 1]), l2norm(dataAblate_10rays[:, 5], dataSimit[:, 1]),
     l2norm(dataAblate_15rays[:, 5], dataSimit[:, 1]), l2norm(dataAblate_25rays[:, 5], dataSimit[:, 1]),
     l2norm(dataAblate_50rays[:, 5], dataSimit[:, 1]),
     l2norm(dataAblate_100rays[:, 5], dataSimit[:, 1]), l2norm(dataAblate_250rays[:, 5], dataSimit[:, 1]),
     l2norm(dataAblate_500rays[:, 5], dataSimit[:, 1])])

plt.figure(figsize=(6, 4), num=4)
plt.loglog(ref_equilibrium, l2_equilibrium, c='black', linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('N', fontsize=10)
plt.ylabel('$\epsilon_T$', fontsize=10)
plt.savefig(savePath + 'Equilibrium_Convergence', dpi=1000, bbox_inches='tight')
plt.show()

# Mesh Convergence Stuff
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
plt.loglog(irrMeshes, l2irrMeshes, c='black', linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)


def minor_formatter(x, pos):
    return f'{int(x)}'


# Set minor formatter for both axes
plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(minor_formatter))
plt.gca().tick_params(axis='x', which='minor', labelsize=8.5)  # set fontsize for minor ticks
plt.ylabel('$\epsilon_T$', fontsize=10)
plt.savefig(savePath + 'Equilibrium_MeshConvergence', dpi=1000, bbox_inches='tight')
# TODO: Change the bounding box to be equal width on both sides of the plot
plt.show()

# %% Results stuff
import numpy as np
import matplotlib.pyplot as plt  # for plotting
import matplotlib.font_manager
import matplotlib.ticker as ticker

# Direct input
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
# Options
W = 5.8
plt.rcParams.update({
    'figure.figsize': (W, W/(4/3)),     # 4:3 aspect ratio
    'font.size' : 10,                   # Set font size to 11pt
    'axes.labelsize': 10,               # -> axis labels
    'legend.fontsize': 10,              # -> legends
    'font.family': 'lmodern',
    'text.usetex': True,
    'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage{lmodern}'
        # ... more packages if needed
    )
})
plt.rc('mathtext', fontset='dejavuserif')
frac_thing = 0.2

savePath = "/home/owen/CLionProjects/ParallelRadiationJCP/figures/"

fig = plt.figure()
fig.set_size_inches(3.54, 3.54)

x = np.linspace(-0.01, 0.01, 1000)
temp = np.zeros(1000)
for i in range(len(x)):
    if x[i] < 0:
        temp[i] = (-(7E6 * x[i] * x[i]) + 2000.0)
    else:
        temp[i] = (-(13E6 * x[i] * x[i]) + 2000.0)

plt.plot(x, temp, c='black', linewidth=0.5)  # , c='blue', s=4)

# Get the current axes, gca stands for 'get current axis'
ax = plt.gca()

# Set the linewidth
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)

plt.xlabel('x $[m]$')  # , fontsize=10)
plt.ylabel(r'T $[K]$')  # , fontsize=10)
plt.ticklabel_format(style="sci")
# plt.subplots_adjust(left=frac_thing, top=1 - frac_thing, right=1 - frac_thing, bottom=frac_thing)
plt.savefig(savePath + 'PlatesTemperature' + '.pdf', dpi=1000, bbox_inches='tight')  # , pad_inches=0)
# TODO: Change the bounding box to be equal width on both sides of the plot
plt.show()
