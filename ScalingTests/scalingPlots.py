import xml.etree.ElementTree as ET
import numpy as np  # for matrix manipulation
import matplotlib.pyplot as plt  # for plotting
import itertools
from os.path import exists

# Set up options that must be defined by the user
# Define the arrays that contain the options which were used
processes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
faces = ["[105, 15]", "[149, 21]", "[210, 30]", "[297, 24]", "[420, 60]", "[594, 85]", "[840, 120]"]
rays = [10, 25, 50, 200]
dims = "2D"

# Template path: "outputs/Scaling2D_30_16_[105, 15].xml"
basePath = "outputs/scalingTests"
initName = "Radiation Initialization"
solveName = "Radiation Solve"

# Define an iterator which stores input parameters and iterates through all combinations of the options
options = itertools.product(rays, processes, faces)

# Problem size doubles for each increase
cells = np.array([[105, 15], [149, 21], [210, 30], [297, 24], [420, 60], [594, 85], [840, 120]])
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
            path = basePath + dims + "_" + str(rays[r]) + "_" + str(processes[p]) + "_" + str(
                faces[f]) + ".xml"  # File path
            path = "outputs/scalingTests2D_10_1_[105, 15].xml"  # TODO Hack for testing
            if exists(path):  # Make sure not to try accessing a path that doesn't exist
                tree = ET.parse(path)  # Create element tree object
                root = tree.getroot()  # Get root element
                # Iterate items (the type of event that contains the data)
                for item in root.findall('./petscroot/selftimertable/event'):
                    # Get the specific name of the event that is desired
                    #    Get the sub-value that is desired out of the event
                    if item.find("name").text == "Radiation Initialization":
                        initTime[r, p, f] = item.find('time/avgvalue').text
                    if item.find("name").text == "Radiation Solve":
                        solveTime[r, p, f] = item.find('time/avgvalue').text

d = 0
# Static scaling analysis
plt.figure(figsize=(6, 4), num=1)
plt.title("Initialization Static Scaling", pad=1)
for i in range(len(processes)):
    plt.loglog(cellsize[d, :], initTime[0, i, :], linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel(r'Time $[s]$', fontsize=10)
plt.ylabel(r'Performance $[\frac{DOF}{s}]$', fontsize=10)
plt.legend(processes, loc="upper left")
plt.savefig('scalingStatic' + dims, dpi=1500, bbox_inches='tight')
plt.show()

# Strong scaling analysis
plt.figure(figsize=(6, 4), num=1)
plt.title("Initialization Strong Scaling", pad=1)
for i in range(len(faces)):
    plt.loglog(processes, initTime[0, :, i], linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel(r'Processes $[s]$', fontsize=10)
plt.ylabel(r'Time $[s]$', fontsize=10)
plt.legend(faces, loc="upper left")
plt.savefig('scalingStrong' + dims, dpi=1500, bbox_inches='tight')
plt.show()

# Weak scaling analysis
# Strong scaling analysis
plt.figure(figsize=(6, 4), num=1)
plt.title("Initialization Strong Scaling", pad=1)
for i in range(len(faces)):
    plt.loglog(processes, initTime[0, :, i], linewidth=1, marker='.')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel(r'Processes $[s]$', fontsize=10)
plt.ylabel(r'Performance $[\frac{DOF}{s}]$', fontsize=10)
plt.legend(faces, loc="upper left")
plt.savefig('scalingWeak' + dims, dpi=1500, bbox_inches='tight')
plt.show()
