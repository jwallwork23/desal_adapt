from desal_adapt.plotting import *
from desal_adapt.parse import Parser
from thetis.utility import create_directory
import numpy as np
import pandas as pd


def read_csv(approach, space='cg1'):
    fname = os.path.join(root_dir, approach, space, 'convergence.log')
    try:
        data = pd.read_csv(fname)
    except FileNotFoundError:
        print(f'File "{fname}" does not exist.')
        return
    return {key: np.array(value) for key, value in data.items()}


# Parse arguments
parser = Parser(prog='test_cases/point_discharge2d/plot_convergence.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-family', 'cg')
parsed_args = parser.parse_args()
config = parsed_args.configuration
assert config in ['aligned', 'offset']
family = parsed_args.family
assert family in ['cg', 'dg']
plot_dir = create_directory(os.path.join('plots', config, f'{family}1'))

# Load data
root_dir = os.path.join('outputs', config)
uniform = read_csv('fixed_mesh')
tags = ['hessian', 'isotropic_dwr', 'anisotropic_dwr', 'weighted_hessian', 'weighted_gradient']
names = ['Hessian-based', 'Isotropic DWR', 'Anisotropic DWR', 'Weighted Hessian', 'Weighted gradient']
markers = ['*', '^', 'V', 'o', 'h']
runs = []
labels = []
for tag, label in zip(tags, names):
    data = read_csv(tag)
    if data is not None:
        runs.append(data)
        labels.append(label)

# QoI errors
truth = uniform['qois'][-1]
uniform['error'] = np.abs((uniform['qois'][:-1] - truth)/truth)
for data in runs:
    data['error'] = np.abs((data['qois'] - truth)/truth)

# Plot QoI convergence vs DoFs
fig, axes = plt.subplots()
axes.semilogx(uniform['dofs'], uniform['qois'], '--', marker='x', label='Uniform')
for data, label, marker in zip(runs, labels, markers):
    axes.semilogx(data['dofs'], data['qois'], '--', marker=marker, label=label)
axes.set_xlabel('DoF count')
axes.set_ylabel('Quantity of interest')
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'dofs_vs_qoi.jpg'))

# Plot QoI error convergence vs DoFs
fig, axes = plt.subplots()
axes.loglog(uniform['dofs'][:-1], uniform['error'], '--', marker='x', label='Uniform')
for data, label, marker in zip(runs, labels, markers):
    axes.loglog(data['dofs'], data['error'], '--', marker=marker, label=label)
axes.set_xlabel('DoF count')
axes.set_ylabel('Relative QoI error')
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'dofs_vs_qoi_error.jpg'))

# Plot QoI error convergence vs wallclock
fig, axes = plt.subplots()
axes.loglog(uniform['wallclock'][:-1], uniform['error'], '--', marker='x', label='Uniform')
for i, (data, label, marker) in enumerate(zip(runs, labels, markers)):
    axes.loglog(data['wallclock'], data['error'], '--', marker=marker, label=label)
    for wc, err, it in zip(data['wallclock'], data['error'], data['iterations']):
        axes.annotate(it, (wc, err), color=f'C{i+1}')
axes.set_xlabel(r'CPU time [$\mathrm s$]')
axes.set_ylabel('Relative QoI error')
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'time_vs_qoi_error.jpg'))

# Plot legend
fig2, axes2 = plt.subplots()
legend = axes2.legend(*axes.get_legend_handles_labels(), fontsize=18, frameon=False)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
plt.savefig(os.path.join(plot_dir, 'legend.jpg'), bbox_inches=bbox)
