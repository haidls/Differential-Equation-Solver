import pde
import numpy
import os

"""grid = pde.CartesianGrid([[0, 1]], [15])
expression = "sin(2*pi*x)+cos(2*pi*x)+2*x+3"
field = pde.ScalarField.from_expression(grid, expression)
boundary_conditions = {"value_expression": expression}
eq = pde.PDE({"u": "laplace(u)/40"}, bc=boundary_conditions)
storage = pde.MemoryStorage()
result = eq.solve(field, t_range=1, dt=1e-1, tracker=storage.tracker(0.1))
pde.plot_kymograph(storage)
print(storage.data)
print(grid.cell_coords)"""

input_length = 2
input_width = 1
test_data_amount = 10
test_repetitions = 1

file_path = 'data_sets/data_set_lwdr_%d_%d_%d_%d.npy' % (input_length, input_width, test_data_amount,
                                                             test_repetitions)
if os.path.exists(file_path):
    print("noice")