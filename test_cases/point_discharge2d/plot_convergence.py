from desal_adapt.plotting import *
from desal_adapt.parse import Parser
from thetis.utility import create_directory
import numpy as np
import pandas as pd


def read_csv(approach, space='cg1'):
    data = pd.read_csv(os.path.join(root_dir, approach, space, 'convergence.log'))
    return {key: np.array(value) for key, value in data.items()}


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
uniform = read_csv('fixed_mesh')
hessian = read_csv('hessian')
truth = uniform['qois'][-1]
uniform['error'] = np.abs((uniform['qois'][:-1] - truth)/truth)
hessian['error'] = np.abs((hessian['qois'] - truth)/truth)

# Plot QoI convergence vs DoFs
fig, axes = plt.subplots()
axes.semilogx(uniform['dofs'], uniform['qois'], '--', marker='x', label='Uniform')
axes.semilogx(hessian['dofs'], hessian['qois'], '--', marker='x', label='Hessian-based')
axes.set_xlabel('DoF count')
axes.set_ylabel('Quantity of interest')
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'dofs_vs_qoi.jpg'))

# Plot QoI error convergence vs DoFs
fig, axes = plt.subplots()
axes.loglog(uniform['dofs'][:-1], uniform['error'], '--', marker='x', label='Uniform')
axes.loglog(hessian['dofs'], hessian['error'], '--', marker='x', label='Hessian-based')
axes.set_xlabel('DoF count')
axes.set_ylabel('Relative QoI error')
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'dofs_vs_qoi_error.jpg'))

# Plot QoI error convergence vs wallclock
fig, axes = plt.subplots()
axes.loglog(uniform['wallclock'][:-1], uniform['error'], '--', marker='x', label='Uniform')
axes.loglog(hessian['wallclock'], hessian['error'], '--', marker='x', label='Hessian-based')
# TODO: Annotate with iteration counts
axes.set_xlabel(r'CPU time [$\mathrm s$]')
axes.set_ylabel('Relative QoI error')
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'time_vs_qoi_error.jpg'))
