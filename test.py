import pde

grid = pde.CartesianGrid([[0, 1]], [15])
expression = "sin(2*pi*x)+cos(2*pi*x)+2*x+3"
field = pde.ScalarField.from_expression(grid, expression)
boundary_conditions = {"value_expression": expression}
eq = pde.PDE({"u": "laplace(u)/40"}, bc=boundary_conditions)
storage = pde.MemoryStorage()
result = eq.solve(field, t_range=1, dt=1e-4, tracker=storage.tracker(0.1))
pde.plot_kymograph(storage)
print(storage.data)
print(grid.cell_coords)