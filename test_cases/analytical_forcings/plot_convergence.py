from desal_adapt.parse import Parser
from desal_adapt.plotting import *
from thetis.utility import create_directory
import pandas as pd


# Parse arguments
parser = Parser(prog='test_cases/analytical_forcings/plot_convergence.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-family', 'cg')
parser.add_argument('-dpi', 500, help="""
    Dots per inch (default 500).
    """)
parsed_args = parser.parse_args()
config = parsed_args.configuration
assert config in ['aligned', 'offset']
family = parsed_args.family
space = f'{family}1'
assert family in ['cg', 'dg']
dpi = parsed_args.dpi
cwd = os.path.dirname(__file__)

# Load data
uniform = {'dofs': [], 'qoi': [], 'wallclock': []}
for i in range(5):
    fname = os.path.join(cwd, 'outputs', config, 'fixed_mesh', f'{family}1', f'level{i}', 'qoi.log')
    try:
        f = pd.read_csv(fname)
    except IOError:
        print(f"File '{fname}' not found.")
        continue
    for key in uniform:
        uniform[key].append(f[key])

# Plot QoI vs DoF count
plot_dir = create_directory(os.path.join(cwd, 'plots', config, f'{family}1'))
fig, axes = plt.subplots()
axes.semilogx(uniform['dofs'], uniform['qoi'], '--x', label='Fixed mesh')
axes.set_xlabel('DoF count')
axes.set_ylabel('Quanity of interest')
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'qoi_vs_dofs_{config}.jpg'), dpi=dpi)

# Plot legend
fig2, axes2 = plt.subplots()
legend = axes2.legend(*axes.get_legend_handles_labels(), fontsize=18, frameon=False, ncol=3)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
plt.savefig(os.path.join(plot_dir, 'legend.jpg'), bbox_inches=bbox, dpi=dpi)
