from thetis.utility import *
from pyroteus.metric import ramp_complexity


__all__ = ["anisotropic_cell_size_3d", "ramp_complexity"]


@PETSc.Log.EventDecorator('anisotropic_cell_size_3d')
def anisotropic_cell_size_3d(mesh):
    """
    Measure of cell size for anisotropic meshes, as described in
    Micheletti et al. (2003).

    This is used in the SUPG formulation for the 3D tracer model.

    Micheletti, Perotto and Picasso (2003). Stabilized finite
    elements on anisotropic meshes: a priori error estimates for
    the advection-diffusion and the Stokes problems. SIAM Journal
    on Numerical Analysis 41.3: 1131-1162.
    """
    try:
        from firedrake.slate.slac.compiler import PETSC_ARCH
    except ImportError:
        PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
    include_dir = ["%s/include/eigen3" % PETSC_ARCH]

    # Compute cell Jacobian
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    J = Function(P0_ten, name="Cell Jacobian")
    J.interpolate(Jacobian(mesh))

    # Compute minimum eigenvalue
    P0 = FunctionSpace(mesh, "DG", 0)
    min_evalue = Function(P0, name="Minimum eigenvalue")
    kernel_str = """
#include <Eigen/Dense>

using namespace Eigen;

void eigmin(double minEval[1], const double * J_) {

  // Map input onto an Eigen object
  Map<Matrix<double, 3, 3, RowMajor> > J((double *)J_);

  // Compute J^T * J
  Matrix<double, 3, 3, RowMajor> A = J.transpose()*J;

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, 3, 3, RowMajor>> eigensolver(A);
  Vector3d D = eigensolver.eigenvalues();

  // Take the square root
  double lambda1 = sqrt(fabs(D(0)));
  double lambda2 = sqrt(fabs(D(1)));
  double lambda3 = sqrt(fabs(D(2)));

  // Select minimum eigenvalue in modulus
  minEval[0] = fmin(lambda1, lambda2);
  minEval[0] = fmin(minEval[0], lambda3);
}
"""
    kernel = op2.Kernel(kernel_str, 'eigmin', cpp=True, include_dirs=include_dir)
    op2.par_loop(kernel, P0_ten.node_set, min_evalue.dat(op2.RW), J.dat(op2.READ))
    return min_evalue
