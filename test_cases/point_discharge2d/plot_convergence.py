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
dpi = 500

# Load data
root_dir = os.path.join('outputs', config)
uniform = read_csv('fixed_mesh')
tags = ['hessian', 'isotropic_dwr', 'anisotropic_dwr', 'weighted_hessian', 'weighted_gradient']
names = ['Hessian-based', 'Isotropic DWR', 'Anisotropic DWR', 'Weighted Hessian', 'Weighted gradient']
markers = ['*', '^', 'v', 'o', 'h']
runs = []
labels = []
for tag, label in zip(tags, names):
    data = read_csv(tag)
    if data is not None:
        runs.append(data)
        labels.append(label)

# Plot QoI convergence vs DoFs
fig, axes = plt.subplots()
if uniform is not None:
    axes.loglog(uniform['dofs'], uniform['qois'], '--', marker='x', label='Uniform')
for data, label, marker in zip(runs, labels, markers):
    axes.loglog(data['dofs'], data['qois'], '--', marker=marker, label=label)
axes.set_xlabel('DoF count')
axes.set_ylabel('Quantity of interest')
axes.set_xticks([1.0e+03, 1.0e+04, 1.0e+05, 1.0e+06])
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'dofs_vs_qoi_{config}.jpg'), dpi=dpi)
if uniform is None:
    print('Cannot plot errors because fixed mesh benchmarks do not exist yet.')
    sys.exit(0)

# QoI errors
truth = uniform['qois'][-1]
uniform['error'] = 100*np.abs((uniform['qois'][:-1] - truth)/truth)
for data in runs:
    data['error'] = 100*np.abs((data['qois'] - truth)/truth)

# Plot QoI error convergence vs DoFs
fig, axes = plt.subplots()
axes.loglog(uniform['dofs'][:-1], uniform['error'], '--', marker='x', label='Uniform')
for data, label, marker in zip(runs, labels, markers):
    axes.loglog(data['dofs'], data['error'], '--', marker=marker, label=label)
axes.set_xlabel('DoF count')
axes.set_ylabel(r'Relative QoI error (\%)')
axes.set_xticks([1.0e+03, 1.0e+04, 1.0e+05, 1.0e+06])
axes.set_yticks([0.01, 0.1, 1, 10, 100])
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'dofs_vs_qoi_error_{config}.jpg'), dpi=dpi)
fig, axes = plt.subplots()
axes.semilogx(uniform['dofs'][:-1], uniform['error'], '--', marker='x', label='Uniform')
for data, label, marker in zip(runs, labels, markers):
    axes.semilogx(data['dofs'], data['error'], '--', marker=marker, label=label)
axes.set_xlabel('DoF count')
axes.set_ylabel(r'Relative QoI error (\%)')
axes.set_xticks([1.0e+03, 1.0e+04, 1.0e+05, 1.0e+06])
axes.set_yticks(np.linspace(0, 100, 11))
axes.set_ylim([-1, 100])
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'dofs_vs_qoi_error_semilogx_{config}.jpg'), dpi=dpi)

# Plot legend
fig2, axes2 = plt.subplots()
legend = axes2.legend(*axes.get_legend_handles_labels(), fontsize=18, frameon=False, ncol=3)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
plt.savefig(os.path.join(plot_dir, 'legend.jpg'), bbox_inches=bbox, dpi=dpi)

# Plot QoI error convergence vs wallclock
fig, axes = plt.subplots()
axes.loglog(uniform['wallclock'][:-1], uniform['error'], '--', marker='x', label='Uniform')
for i, (data, label, marker) in enumerate(zip(runs, labels, markers)):
    axes.loglog(data['wallclock'], data['error'], '--', marker=marker, label=label)
    for wc, err, it in zip(data['wallclock'], data['error'], data['iterations']):
        axes.annotate(it, (wc, err), color=f'C{i+1}')
axes.set_xlabel(r'CPU time [$\mathrm s$]')
axes.set_ylabel(r'Relative QoI error (\%)')
axes.set_xticks([1.0e+00, 1.0e+01, 1.0e+02, 1.0e+03])
axes.set_yticks([0.01, 0.1, 1, 10, 100])
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'time_vs_qoi_error_{config}.jpg'))
