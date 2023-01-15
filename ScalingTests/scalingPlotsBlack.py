import xml.etree.ElementTree as ET
import numpy as np  # for matrix manipulation
import matplotlib.pyplot as plt  # for plotting
import itertools
from os.path import exists

plt.rcParams["font.family"] = "Noto Serif CJK JP"

# Set up options that must be defined by the user
colorarray = ["black", "grey", "black", "black", "darkorange", "goldenrod", "yellow", "yellowgreen",
              "green", "lightgreen", "teal", "powderblue", "darkorchid", "violet",
              "palevioletred"]
markerarray = [".", "1", "P", "*"]

# Define the arrays that contain the options which were used
processes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
faces = ["[105,15]", "[149,21]", "[297,42]", "[297,42,42]", "[420,60]", "[594,85]", "[840,120]"] #[210,30]
rays = np.array([5, 10, 25, 50])
dtheta = 180 / rays
dims = " 3D"

# Template path: "outputs/Scaling2D_30_16_[105, 15].xml"
basePath = "slabRadSF2DScaling/scalingSFOutputs/"
initName = "Radiation::Initialize"
solveName = "Radiation::EvaluateGains"

# Define an iterator which stores input parameters and iterates through all combinations of the options
options = itertools.product(rays, processes, faces)

# Problem size doubles for each increase
cells = np.array([[105, 15], [149, 21], [210, 30], [297, 42], [420, 60], [594, 85], [840, 120]])
cellsize = np.ones([2, np.shape(cells)[0]])
for n in range(np.shape(cellsize)[0]):
    for i in range(np.shape(cells)[0]):
        for j in range(np.shape(cells)[1]):
            cellsize[n, i] *= cells[i, j]
            if n == 1:
                cellsize[n, i] *= cells[i, 1]

# Create arrays which the parsed information will be stored inside: Whatever information is desired
initTime = np.zeros((len(rays), len(processes), len(faces)))
solveTime = np.zeros((len(rays), len(processes), len(faces)))

# Iterate through the arrays to get information out of the xml files
for r in range(len(rays)):
    for p in range(len(processes)):
        for f in range(len(faces)):
            # Create strings which represent the file names of the outputs
            path = basePath + "slabRadSF2D" + "_" + str(rays[r]) + "_" + str(processes[p]) + "_" + str(
                faces[f]) + ".xml"  # File path
            # path = "outputs/scalingTests2D_10_1_[105, 15].xml"  # Hack for testing
            if exists(path):  # Make sure not to try accessing a path that doesn't exist
                tree = ET.parse(path)  # Create element tree object
                root = tree.getroot()  # Get root element
                # Iterate items (the type of event that contains the data)
                for item in root.findall('./petscroot/selftimertable/event'):
                    # Get the specific name of the event that is desired
                    #    Get the sub-value that is desired out of the event
                    if item.find("name").text == initName:
                        if not item.find('time/maxvalue') is None:
                            initTime[r, p, f] = item.find('time/maxvalue').text
                    if item.find("name").text == solveName:
                        if not item.find('time/maxvalue') is None:
                            solveTime[r, p, f] = item.find('time/maxvalue').text
            if initTime[r, p, f] == 0:
                initTime[r, p, f] = float("nan")
            if solveTime[r, p, f] == 0:
                solveTime[r, p, f] = float("nan")

processes = np.asarray(processes)
faces = np.asarray(faces)
rays = np.asarray(rays)

f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
handles = [f(markerarray[i], "k") for i in range(len(markerarray))]
handles += [f("s", colorarray[i]) for i in range(len(colorarray))]

d = 0
# Initialization static scaling analysis
# plt.figure(figsize=(10, 6), num=1)
# plt.title("Initialization Static Scaling" + dims, pad=1)
# for n in range(len(rays)):
#     for i in range(len(processes)):
#         mask = np.isfinite(initTime[n, i, :])
#         x = cellsize[d, :]
#         y = cellsize[d, :] / initTime[n, i, :]
#         plt.loglog(x[mask], y[mask], linewidth=1, marker=markerarray[n], c=colorarray[i])
# plt.yticks(fontsize=10)
# plt.xticks(fontsize=10)
# plt.xlabel(r'DOF $[cells]$', fontsize=10)
# plt.ylabel(r'Performance $[\frac{DOF}{s}]$', fontsize=10)
# labels = dtheta
# labels = np.append(labels, processes)
# plt.legend(handles, labels, loc="upper left")
# plt.savefig('initScalingStatic' + dims, dpi=1500, bbox_inches='tight')
# plt.show()

# Initialization Strong scaling analysis
plt.figure(figsize=(6, 4), num=2)
# plt.title("Initialization Strong Scaling" + dims, pad=1)
for n in range(len(rays)):
    for i in range(len(faces)):
        mask = np.isfinite(initTime[n, :, i])
        x = processes
        y = initTime[n, :, i]

        # Bring the lowest available index to the line to normalize the scaling plot * (ideal / lowest available index)
        first = np.argmax(mask)

        plt.loglog(x[mask], (processes[first] * y[first]) / y[mask], linewidth=1, marker=markerarray[n],
                   c=colorarray[i])
plt.plot(processes, processes, linewidth=1, c="black", linestyle="--")
plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel(r'MPI Processes', fontsize=10)
plt.ylabel(r'Speedup', fontsize=10)
labels = dtheta
labels = np.append(labels, faces)
# plt.legend(handles, labels, loc="upper left")
plt.savefig('initScalingStrongBlack' + dims, dpi=1500, bbox_inches='tight')
plt.show()

# Initialization Weak scaling analysis
# weakInit = np.zeros([len(rays), len(processes) + len(faces), len(faces)])
# I = len(processes)
# for n in range(len(rays)):
#     for i in range(len(processes)):
#         for j in range(len(faces)):
#             x = ((I-1) - i) + j
#             weakInit[n, x, j] = initTime[n, i, j]
#             if weakInit[n, x, j] == 0:
#                 weakInit[n, x, j] = float("nan")

# plt.figure(figsize=(6, 4), num=1)
# plt.title("Initialization Weak Scaling", pad=1)
# I = len(processes)
# for n in range(len(rays)):
#     for i in range(len(faces) + len(processes)):
#         weakInit = np.diagonal(initTime, offset=(i-I), axis1=0, axis2=1)
#         mask = np.isfinite(weakInit)
#         x = cellsize[d, :]
#         y = cellsize[d, :] / weakInit[:]
#         # Bring the lowest available index to the line to normalize the scaling plot * (ideal / lowest available index)
#         first = np.argmax(mask)
#
#         plt.loglog(x[mask], (cellsize[first] * y[first]) / y[mask], linewidth=1, marker='.')
# plt.yticks(fontsize=10)
# plt.xticks(fontsize=10)
# plt.xlabel(r'DOF $[cells]$', fontsize=10)
# plt.ylabel(r'Efficiency', fontsize=10)
# # plt.legend(faces, loc="upper left")
# plt.savefig('scalingWeak' + dims, dpi=1500, bbox_inches='tight')
# plt.show()


# Solve static scaling analysis
# plt.figure(figsize=(10, 6), num=3)
# plt.title("Solve Static Scaling" + dims, pad=1)
# for n in range(len(rays)):
#     for i in range(len(processes)):
#         mask = np.isfinite(solveTime[n, i, :])
#         x = cellsize[d, :]
#         y = cellsize[d, :] / solveTime[n, i, :]
#         plt.loglog(x[mask], y[mask], linewidth=1, marker=markerarray[n],
#                    c=colorarray[i])
# plt.yticks(fontsize=10)
# plt.xticks(fontsize=10)
# plt.xlabel(r'DOF $[cells]$', fontsize=10)
# plt.ylabel(r'Performance $[\frac{DOF}{s}]$', fontsize=10)
# labels = dtheta
# labels = np.append(labels, processes)
# plt.legend(handles, labels, loc="upper left")
# plt.savefig('solveScalingStatic' + dims, dpi=1500, bbox_inches='tight')
# plt.show()

# Initialization Strong scaling analysis
plt.figure(figsize=(6, 4), num=4)
# plt.title("Solve Strong Scaling" + dims, pad=1)
for n in range(len(rays)):
    for i in range(len(faces)):
        mask = np.isfinite(solveTime[n, :, i])
        x = processes
        y = solveTime[n, :, i]

        # Bring the lowest available index to the line to normalize the scaling plot * (ideal / lowest available index)
        first = np.argmax(mask)

        plt.loglog(x[mask], (processes[first] * y[first]) / y[mask], linewidth=1, marker=markerarray[n],
                   c=colorarray[i])
plt.plot(processes, processes, linewidth=1, c="black", linestyle="--")
plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel(r'MPI Processes', fontsize=10)
plt.ylabel(r'Speedup', fontsize=10)
labels = dtheta
labels = np.append(labels, faces)
# plt.legend(handles, labels, loc="upper left")
plt.savefig('solveScalingStrongBlack' + dims, dpi=1500, bbox_inches='tight')
plt.show()
