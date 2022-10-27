import xml.dom.minidom as xml  # for parsing xml
import xml.etree.ElementTree as ET
import numpy as np  # for matrix manipulation
import matplotlib.pyplot as plt  # for plotting
import itertools
from os.path import exists

# Set up options that must be defined by the user
# Define the arrays that contain the options which were used
processes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
faces = ["[105, 15]", "[140, 20]", "[175, 25]", "[210, 30]", "[350, 50]", "[560, 80]"]
rays = [10, 25, 50, 200]
# dims = ["2D", "3D"]

# Template path: "outputs/Scaling2D_30_16_[105, 15].xml"
basePath = "outputs/scalingTests2D"
initName = "Radiation Initialization"
solveName = "Radiation Solve"

# Define an iterator which stores input parameters and iterates through all combinations of the options
options = itertools.product(rays, processes, faces)

# Code that should remain constant
# Create arrays which the parsed information will be stored inside
initTime = np.zeros((len(rays), len(processes), len(faces)))
solveTime = np.zeros((len(rays), len(processes), len(faces)))

# Iterate through the arrays to get the scaling information out of the xml files
# for i in options:
for r in range(len(rays)):
    for p in range(len(processes)):
        for f in range(len(faces)):
            # Create strings which represent the file names of the outputs
            # ray, proc, face = i  # Get the unique identifier
            path = basePath + "_" + str(rays[r]) + "_" + str(processes[p]) + "_" + str(faces[f]) + ".xml"  # File path
            path = "outputs/scalingTests2D_10_1_[105, 15].xml" # Hack for testing
            if exists(path):
                # doc = xml.parse(path)  # Get the specified file as an output
                tree = ET.parse(path) # create element tree object
                root = tree.getroot() # get root element
                # iterate news items
                for item in root.findall('./petscroot/timertree/event'):
                    # iterate child elements of item
                    for child in item:
                        if child.name == '{http://search.yahoo.com/mrss/}content':
                            news['media'] = child.attrib['url']

                    # append news dictionary to news items list
                    items.append(news)

                initTime[r, p, f] = #doc.getElementsByTagName(initName)  # Get the information out of the xml file
                solveTime[r, p, f] = #doc.getElementsByTagName(solveName)  # Get the desired information out of the xml file

# After the values have been stored, they can be plotted
plt.figure(figsize=(6, 4), num=1)
plt.title("Initialization Static Scaling", pad=1)
plt.plot(initTime[0, 0, :], linewidth=1)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel(r'Time $[s]$', fontsize=10)
plt.ylabel(r'Performance $[\frac{DOF}{s}]$', fontsize=10)
plt.savefig('scalingTests', dpi=1500, bbox_inches='tight')
plt.show()
