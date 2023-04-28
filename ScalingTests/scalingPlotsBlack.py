import xml.etree.cElementTree as ET
import numpy as np  # for matrix manipulation
import matplotlib.pyplot as plt  # for plotting
import itertools
from os.path import exists
from scipy.optimize import curve_fit
import argparse
import pathlib


# Template path: "outputs/Scaling2D_30_16_[105, 15].xml"
# basePath = "slabRadSF2DScaling/scalingCsv/volumetricCsv/"
# basePath = "csvFiles/"
# initName = b"Radiation::Initialize"
# solveName = b"Radiation::EvaluateGains"

class PerformanceAnalysis:

    def __init__(self, base_path=None, name=None, processes=None, faces=None, cellsize=None, rays=None, events=None):
        self.basePath = base_path
        self.name = name
        self.processes = processes
        self.faces = faces
        self.cellsize = cellsize
        self.events = events
        self.rays = rays

    @staticmethod
    def r_squared_func(y, y_fit):
        return 1 - np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def load_csv_files(self):
        # Create arrays which the parsed information will be stored inside: Whatever information is desired
        times = np.zeros(len(self.events), len(self.rays), len(self.processes), len(self.faces))
        # Iterate through the arrays to get information out of the files
        for r in range(len(self.rays)):
            for p in range(len(self.processes)):
                for f in range(len(self.faces)):
                    # Create strings which represent the file names of the outputs
                    path = self.base_path + self.name + "_" + str(self.rays[r]) + "_" + str(
                        self.processes[p]) + "_" + str(
                        self.faces[f]) + ".csv"  # File path
                    dtypes = {'names': ('stage', 'name', 'time'),
                              'formats': ('S30', 'S30', 'f4')}

                    if exists(path):  # If the path exists then it can be loaded
                        data = np.loadtxt(path, delimiter=",", dtype=dtypes, skiprows=1, usecols=(0, 1, 4))
                        lines = len(data)  # Get the length of the csv
                        for e in range(len(self.events)):

                            for i in range(lines):  # Iterate through all the lines in the csv
                                if (data[i][1] == self.events[e]) and (
                                        data[i][2] > times[e, r, p, f]):  # Check current line
                                    times[e, r, p, f] = data[i][
                                        2]  # If it is, then write the value in column 4 of that line

                            # If the time is never written then the filter code then don't try to plot it
                            if times[e, r, p, f] == 0:
                                times[e, r, p, f] = float("nan")

        self.processes = np.asarray(self.processes)
        self.faces = np.asarray(self.faces)
        self.rays = np.asarray(self.rays)

    # Do curve fitting of the data for performance modelling purposes.
    @staticmethod
    def gustafson_func(self, N, t0, s, c, d, f):
        # intensity = s / N
        # performance = np.min(N, d)
        # return t0 * (N ** f) + c * performance * np.log10(N)
        # return t0 * (1 / ((N ** c) * performance) +  * np.log(N))
        # return t0 * (1 / ((N ** c) * s) + (1 - 1 / (N ** c)) * np.log(N))
        return t0 * np.log(N) + s * np.log(N) * np.log(N) + d * np.log(N) * np.log(N) * np.log(N) + c * N ** f

    # Set bounds for the parameters (all non-negative)
    # param_bounds = ([0, -np.inf, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
    param_bounds = ([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])

    def plot_static_scaling(self):
        for e in range(len(self.events)):
            # Initialization static scaling analysis
            plt.figure(figsize=(10, 6), num=1)
            plt.title("Initialization Static Scaling" + dims, pad=1)
            for n in range(len(rays)):
                for i in range(len(self.processes)):
                    mask = np.isfinite(self.times[e, n, i, :])
                    x = self.cellsize
                    y = self.cellsize / self.times[e, n, i, :]
                    plt.loglog(x[mask], y[mask], linewidth=1, marker=markerarray[n], c=colorarray[i])
            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)
            plt.xlabel(r'DOF $[cells]$', fontsize=10)
            plt.ylabel(r'Performance $[\frac{DOF}{s}]$', fontsize=10)
            labels = dtheta
            labels = np.append(labels, self.processes)
            plt.legend(handles, labels, loc="upper left")
            plt.savefig('StaticScaling_' + self.events[e] + ".png", dpi=1500, bbox_inches='tight')
            plt.show()

    def plot_weak_scaling(self):
        for e in range(len(self.events)):
            # Initialization Weak scaling analysis
            weak_init = np.zeros([len(self.rays), len(self.processes) + len(self.faces), len(self.faces)])
            I = len(self.processes)
            for n in range(len(rays)):
                for i in range(len(self.processes)):
                    for j in range(len(self.faces)):
                        x = ((I - 1) - i) + j
                        weak_init[n, x, j] = self.times[e, n, i, j]
                        if weak_init[n, x, j] == 0:
                            weak_init[n, x, j] = float("nan")

            plt.figure(figsize=(6, 4), num=1)
            plt.title("Initialization Weak Scaling", pad=1)
            I = len(self.processes)
            for n in range(len(rays)):
                for i in range(len(self.faces) + len(self.processes)):
                    weak_init = np.diagonal(self.times, offset=(i - I), axis1=0, axis2=1)
                    mask = np.isfinite(weak_init)
                    x = self.cellsize
                    y = self.cellsize / weak_init
                    # Bring the lowest available index to the line to normalize the scaling plot *
                    # (ideal / lowest available index)
                    first = np.argmax(mask)

                    plt.loglog(x[mask], (self.cellsize[first] * y[first]) / y[mask], linewidth=1, marker='.')
            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)
            plt.xlabel(r'DOF $[cells]$', fontsize=10)
            plt.ylabel(r'Efficiency', fontsize=10)
            # plt.legend(faces, loc="upper left")
            plt.savefig('WeakScaling_' + self.events[e] + ".png", dpi=1500, bbox_inches='tight')
            plt.show()

    def plot_strong_scaling(self, function_fit):
        # Initialization Strong scaling analysis
        plt.figure(figsize=(6, 4), num=4)
        # plt.title("Solve Strong Scaling" + dims, pad=1)
        for e in range(len(self.events)):
            for n in range(len(rays)):
                for i in range(len(self.faces)):
                    mask = np.isfinite(self.times[e, n, :, i])
                    x = self.processes
                    y = self.times[e, n, :, i]

                    # Bring the lowest available index to the line to normalize the scaling plot *
                    # (ideal / lowest available index)
                    first = np.argmax(mask)

                    if np.any(mask):
                        # Perform non-linear curve fitting
                        popt, pcov = curve_fit(function_fit, x[mask], y[mask])  # , bounds=param_bounds)

                        # Extract the fitted parameters
                        t0, s, c, d, f = popt

                        # Calculate the R-squared value
                        y_fit = function_fit(x[mask], t0, s, c, d, f)
                        r_squared = self.r_squared_func(y[mask], y_fit)

                        plt.loglog(x[mask], (self.processes[first] * y[first]) / y_fit, c="black", linestyle="-.",
                                   label=f'Fitted curve: t0={t0:.2f}, s={s:.2f}, c={c:.2f}, r^2={r_squared:.2f}')
                        print(f't0={t0:.2f}, s={s:.2f}, c={c:.2f}, d={d:.2f}, f={f:.2f}, r^2={r_squared:.2f}')
                    adjusted = (self.processes[first] * y[first]) / y[mask]
                    plt.loglog(x[mask], adjusted, linewidth=1, marker=colorarray[i],
                               c="black", markersize=4)
            plt.plot(self.processes, self.processes, linewidth=1, c="black", linestyle="--")
            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)
            plt.xlabel(r'MPI Processes', fontsize=10)
            plt.ylabel(r'Speedup', fontsize=10)
            # labels = dtheta
            # labels = np.append(labels, faces)
            # plt.legend(["2D"], loc="upper left")  # , "3D"
            # plt.legend()
            plt.savefig('StrongScaling_' + self.events[e] + ".png", dpi=1500, bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process scaling data from PETSc events.')
    parser.add_argument('--path', dest='base_path', type=pathlib.Path, required=True,
                        help='Path to the base directory containing scaling csv files.')
    parser.add_argument('--name', dest='name', type=str, required=True,
                        help='Title of the files being processed.')
    parser.add_argument('--processes', dest='processes', type=int, required=True, nargs="+",
                        help='Process counts being considered.')
    parser.add_argument('--problems', dest='problems', type=str, required=True, nargs="+",
                        help='Problems being considered.')
    parser.add_argument('--dofs', dest='cellsize', type=float, required=True, nargs="+",
                        help='Mesh or dof size associated with each problem.')
    parser.add_argument('--events', dest='events', type=str, required=True, nargs="+",
                        help='Event names to measure.')

    args = parser.parse_args()
    rays = np.array([5, 10, 25, 50])

    scaling_data = PerformanceAnalysis(args.base_path, args.name, args.processes,
                                       args.problems, args.cellsize, rays, args.events)

    # Set up plotting options that must be defined by the user
    colorarray = ["o", "s", "o", "s", ".", ".", ".", ".",
                  ".", ".", ".", ".", ".", ".",
                  "."]
    markerarray = [".", "1", "P", "*"]
    plt.rcParams["font.family"] = "Noto Serif CJK JP"

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f(markerarray[i], "k") for i in range(len(markerarray))]
    handles += [f("s", "black") for i in range(len(colorarray))]

    # Define the arrays that contain the options which were used
    # processes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]  # , 8192, 16384, 32768]
    # faces = ["[105,15]", "[149,21]", "[297,42]"]  # , "[297,42,42]", "[420,60]", "[594,85]", "[594,85,85]",
    # "[840,120]"]  # [210,30]
    dtheta = 180 / rays
    dims = "_vol"
