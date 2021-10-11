from __future__ import absolute_import
from thetis.utility import *
from thetis.tracer_eq_2d import *


__all__ = [
    'TracerEquation3D',
]


class HorizontalAdvectionTerm(TracerTerm):
    # TODO: docstring
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions):
        if fields_old.get('uv_3d') is None:
            return 0
        elev = fields_old['elev_3d']
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor * fields_old['uv_3d']
        # FIXME is this an option?
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        f = 0
        f += -(Dx(uv[0] * self.test, 0) * solution
               + Dx(uv[1] * self.test, 1) * solution
               + Dx(uv[2] * self.test, 2) * solution) * self.dx

        if self.horizontal_dg:
            # add interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0]
                     + uv_av[1]*self.normal('-')[1]
                     + uv_av[2]*self.normal('-')[2])
            s = 0.5*(sign(un_av) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)

            f += c_up*(jump(self.test, uv[0] * self.normal[0])
                       + jump(self.test, uv[1] * self.normal[1])
                       + jump(self.test, uv[2] * self.normal[2])) * self.dS
            # Lax-Friedrichs stabilization
            if self.options.use_lax_friedrichs_tracer:
                gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                f += gamma*dot(jump(self.test), jump(solution))*self.dS

        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            c_in = solution
            if funcs is not None:
                c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                uv_av = 0.5*(uv + uv_ext)
                un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1] + self.normal[2]*uv_av[2]
                s = 0.5*(sign(un_av) + 1.0)
                c_up = c_in*s + c_ext*(1-s)
                f += c_up*(uv_av[0]*self.normal[0]
                           + uv_av[1]*self.normal[1]
                           + uv_av[2]*self.normal[2])*self.test*ds_bnd
            else:
                f += c_in * (uv[0]*self.normal[0]
                             + uv[1]*self.normal[1]
                             + uv[2]*self.normal[2])*self.test*ds_bnd

        return -f


class HorizontalDiffusionTerm(TracerTerm):
    # TODO: docstring
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, 0, ],
                                 [0, diffusivity_h, 0, ],
                                 [0, 0, diffusivity_h, ]])
        grad_test = grad(self.test)
        diff_flux = dot(diff_tensor, grad(solution))
        sipg_factor = self.options.sipg_factor_tracer

        f = 0
        f += inner(grad_test, diff_flux)*self.dx

        if self.horizontal_dg:
            cell = self.mesh.ufl_cell()
            p = self.function_space.ufl_element().degree()
            cp = (p + 1) * (p + 2) / 2 if cell == triangle else (p + 1)**2
            l_normal = CellVolume(self.mesh) / FacetArea(self.mesh)
            # by default the factor is multiplied by 2 to ensure convergence
            sigma = sipg_factor * cp / l_normal
            sp = sigma('+')
            sm = sigma('-')
            sigma_max = conditional(sp > sm, sp, sm)
            ds_interior = self.dS
            f += sigma_max * inner(
                jump(self.test, self.normal),
                dot(avg(diff_tensor), jump(solution, self.normal)))*ds_interior
            f += -inner(avg(dot(diff_tensor, grad(self.test))),
                        jump(solution, self.normal))*ds_interior
            f += -inner(jump(self.test, self.normal),
                        avg(dot(diff_tensor, grad(solution))))*ds_interior

        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            c_in = solution
            elev = fields_old['elev_3d']
            self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
            uv = self.corr_factor * fields_old['uv_3d']
            if funcs is not None:
                if 'diff_flux' in funcs:
                    f += -self.test*funcs['diff_flux']*ds_bnd
                else:
                    c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                    uv_av = 0.5*(uv + uv_ext)
                    un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1] + self.normal[2]*uv_av[2]
                    s = 0.5*(sign(un_av) + 1.0)
                    c_up = c_in*s + c_ext*(1-s)
                    diff_flux_up = dot(diff_tensor, grad(c_up))
                    f += -self.test*dot(diff_flux_up, self.normal)*ds_bnd

        return -f


class TracerEquation3D(TracerEquation2D):
    """
    Tracer equation for 3D problems
    """
    def add_terms(self, *args, **kwargs):
        self.add_term(HorizontalAdvectionTerm(*args, **kwargs), 'explicit')
        self.add_term(HorizontalDiffusionTerm(*args, **kwargs), 'explicit')
        self.add_term(SourceTerm(*args, **kwargs), 'source')
