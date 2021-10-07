from desal_adapt.plotting import *
from desal_adapt.parse import Parser
from thetis.utility import create_directory
import numpy as np
import pandas as pd


# Parse arguments
parser = Parser(prog='test_cases/point_discharge2d/plot_convergence.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parsed_args = parser.parse_args()
config = parsed_args.configuration
assert config in ['aligned', 'offset']
plot_dir = create_directory(os.path.join('plots', config, 'cg1'))

# Load data
root_dir = os.path.join('outputs', config)
uniform = pd.read_csv(os.path.join(root_dir, 'fixed_mesh', 'cg1', 'convergence.log'))
uniform = {key: np.array(value) for key, value in uniform.items()}
truth = uniform['qois'][-1]
uniform['error'] = np.abs((uniform['qois'][:-1] - truth)/truth)

# Plot QoI convergence vs DoFs
fig, axes = plt.subplots()
axes.semilogx(uniform['dofs'], uniform['qois'], '--', marker='x', label='Uniform')
axes.set_xlabel('DoF count')
axes.set_ylabel('Quantity of interest')
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'dofs_vs_qoi.jpg'))

# Plot QoI error convergence vs DoFs
fig, axes = plt.subplots()
axes.loglog(uniform['dofs'][:-1], uniform['error'], '--', marker='x', label='Uniform')
axes.set_xlabel('DoF count')
axes.set_ylabel('Relative QoI error')
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'dofs_vs_qoi_error.jpg'))

# Plot QoI error convergence vs wallclock
fig, axes = plt.subplots()
axes.loglog(uniform['wallclock'][:-1], uniform['error'], '--', marker='x', label='Uniform')
axes.set_xlabel(r'CPU time [$\mathrm s$]')
axes.set_ylabel('Relative QoI error')
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'time_vs_qoi_error.jpg'))
