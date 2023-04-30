import numpy as np

n = 20
x_array = np.zeros(n)
y_array = np.zeros(n)
cell_array = np.zeros(n)

x_array[0] = 15
y_array[0] = 7 * x_array[0]

cell_array = x_array * y_array
for i in range(len(cell_array) - 1):
    cell_array[i + 1] = cell_array[i] * 2
print(cell_array)

# 2D mesh size calculation
# area = x**2 * 7
# np.sqrt(cells / 7) = x
x_array = np.sqrt(cell_array / 7.0)
y_array = x_array * 7
print(x_array)
print(y_array)

# 3D mesh size calculation
# volume = x**3 * 7
# np.cbrt(cells / 7) = x
x_array = np.cbrt(cell_array / 7.0)
y_array = x_array * 7
print(x_array)
print(y_array)

# Groups
# Faces[min, max] Processes [min, max]
groups = np.array([
    [0.0E0, 6.0E6, 1.0E0, 2.3E3],
    [4.6E4, 4.6E7, 2.3E3, 1.8E4],
    [3.7E5, 1.8E8, 1.8E4, 7.4E4],
    [5.5E6, 7.4E8, 7.7E4, 2.9E5]
])
