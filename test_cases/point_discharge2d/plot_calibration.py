from desal_adapt.parse import Parser
from desal_adapt.plotting import *
import numpy as np


# Parse arguments
parser = Parser(prog='test_cases/point_discharge2d/plot_calibration.py')
parser.add_argument('-level', 5, help="""
    Mesh resolution level (default 5).
    """)
parsed_args = parser.parse_args()
level = parsed_args.level

# Load data
m = np.load(f'data/control_progress_{level}.npy')
J = np.load(f'data/func_progress_{level}.npy')

# Plot
fig, axes = plt.subplots()
axes.semilogy(m*100, J, '-x')
axes.set_xlabel(r'Radius [$\mathrm{cm}$]')
axes.set_ylabel(r'$L^2$ error')
plt.tight_layout()
plt.savefig(f'plots/calibration_{level}.jpg', dpi=500)
