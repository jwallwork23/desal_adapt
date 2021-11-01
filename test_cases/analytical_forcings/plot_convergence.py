from desal_adapt.parse import Parser
from desal_adapt.plotting import *
from thetis.utility import create_directory
import numpy as np
import pandas as pd


# Parse arguments
parser = Parser(prog='test_cases/analytical_forcings/plot_convergence.py')
parser.add_argument('-configuration', 'offset', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-family', 'cg')
parsed_args = parser.parse_args()
config = parsed_args.configuration
assert config in ['aligned', 'offset']
family = parsed_args.family
space = f'{family}1'
assert family in ['cg', 'dg']
cwd = os.path.dirname(__file__)
root_dir = os.path.join(cwd, 'outputs', config)

# Load data
approaches = ['fixed_mesh', 'isotropic_dwr', 'anisotropic_dwr', 'weighted_hessian', 'weighted_gradient']
names = ['Fixed mesh', 'Isotropic DWR', 'Anisotropic DWR', 'Weighted Hessian', 'Weighted gradient']
markers = ['x', '^', 'v', 'o', 'h']
colours = ['C0', 'C2', 'C3', 'C4', 'C5']
data = {approach: {'dofs': [], 'qoi': [], 'wallclock': []} for approach in approaches}
for approach in approaches:
    for i in range(5):
        if approach == 'fixed_mesh':
            # fname = os.path.join(root_dir, approach, f'{family}1', f'levels{i}', 'refs0', 'qoi.log')
            fname = os.path.join(root_dir, approach, f'{family}1', 'levels0', f'refs{i}', 'qoi.log')
            if not os.path.exists(fname):
                print(f"file '{fname}' not found.")
                continue
            f = pd.read_csv(fname)
            for key in data[approach]:
                data[approach][key].append(f[key][0])
        else:
            fname = os.path.join(root_dir, approach, f'{family}1', f'target{1000*4**i}', 'qoi.log')
            if not os.path.exists(fname):
                print(f"File '{fname}' not found.")
                continue
            lines = open(fname, 'r').readlines()
            data[approach]['qoi'].append(float(lines[1].split('=')[-1]))
            data[approach]['wallclock'].append(float(lines[2].split('=')[-1].split()[0]))

# Plot QoI vs DoF count
plot_dir = create_directory(os.path.join(cwd, 'plots', config, f'{family}1'))
fig, axes = plt.subplots()
for approach, name, marker, colour in zip(approaches, names, markers, colours):
    if approach != 'fixed_mesh':
        continue
    axes.semilogx(data[approach]['dofs'], data[approach]['qoi'],
                  linestyle='--', marker=marker, label=name, color=colour)
axes.set_xlabel('DoF count')
axes.set_ylabel('Quantity of Interest')
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'qoi_vs_dofs_{config}.pdf'))

# Plot QoI vs CPU time
fig, axes = plt.subplots()
for approach, name, marker, colour in zip(approaches, names, markers, colours):
    axes.plot(data[approach]['wallclock'], data[approach]['qoi'],
              linestyle='--', marker=marker, label=name, color=colour)
axes.set_xlabel(r'CPU time [$\mathrm s$]')
axes.set_ylabel('Quantity of Interest')
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'qoi_vs_time_{config}.pdf'))

# Plot QoI vs level
fig, axes = plt.subplots()
for approach, name, marker, colour in zip(approaches, names, markers, colours):
    if approach == 'fixed_mesh':
        continue
    N = len(data[approach]['qoi'])
    axes.semilogx(2.0e+06*4.0**np.array(range(N)), data[approach]['qoi'],
                  linestyle='--', marker=marker, label=name, color=colour)
axes.set_xlabel('Target space-time complexity')
axes.set_ylabel('Quantity of Interest')
axes.set_xticks([1.0e+06, 1.0e+07, 1.0e+08])
axes.set_yticks([9.275e+06, 9.3e+06, 9.325e+06, 9.350e+06, 9.375e+06])
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'qoi_vs_target_{config}.pdf'))

# Plot legend
fig2, axes2 = plt.subplots()
legend = axes2.legend(*axes.get_legend_handles_labels(), fontsize=18, frameon=False, ncol=1)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
plt.savefig(os.path.join(plot_dir, 'legend.pdf'), bbox_inches=bbox)
