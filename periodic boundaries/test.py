import pde
import numpy
import os
import matplotlib.pyplot as plt
import time

grid = pde.CartesianGrid([[0, 1]], [50], periodic=True)
expression = "cos(2*pi*x)**3+cos(4*pi*x)**3+sin(2*pi*x)**3+3"
field = pde.ScalarField.from_expression(grid, expression)
boundary_conditions = {"periodic"}
eq = pde.PDE({"φ": "(6 * φ * d_dx(φ) - laplace(d_dx(φ)))/10000"}, bc=boundary_conditions) #"u": "d_dx(10*u+4*d_dx(u))"
storage = pde.MemoryStorage()
result = eq.solve(field, t_range=1, dt=1e-7, tracker=storage.tracker(0.1))
pde.plot_kymograph(storage)
print(storage.data)
print(grid.cell_coords)

"""array_3d = numpy.zeros((8, 4, 2))
for i in range(0,64):
    array_3d[(i//8)%8, (i//2)%4, i%2] = i
shape = (8, 8)
array_2d = numpy.reshape(array_3d, shape)
print(array_3d)
print(array_2d)

array_1d = array_2d.flatten()
print(array_1d)

list = []
list.append(numpy.array([[1, 2], [3, 4]]))
list.append(numpy.array([[5, 6], [7, 8]]))

list_array = numpy.stack(list)
print(list_array)"""

