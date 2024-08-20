#include "core/math.hpp"
#include "core/random.h"
// #include "data/curand_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/physics/lorentz_transform.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/kernel_helper.hpp"

using namespace std;
namespace Aperture {


template <typename Conf>
void
kink_deriven(vector_field<Conf> &B, particle_data_t &ptc,
                     rng_states_t<exec_tags::device> &states) {
  using value_t = typename Conf::value_t;
  //value_t B_z = sim_env().params().get_as<double>("B_z_ratio", 0.0);
  //value_t sigma = sim_env().params().get_as<double>("sigma", 4.0);
  value_t R = sim_env().params().get_as<double>("Radius", 1.0);
  value_t kT = sim_env().params().get_as<double>("Temperature", 0.01);

  value_t n = sim_env().params().get_as<double>("number_density", 1000.0);
  //value_t n_c = sim_env().params().get_as<double>("core_n", 2000.0);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);
  value_t ppc = sim_env().params().get_as<double>("ppc", 8.0);
  

  
  value_t B0 = 100.0;
  auto &grid = B.grid();
  auto ext = grid.extent();

  // Initialize the magnetic field values
  B.set_values(0, [B0](auto x, auto y, auto z) {
    return -B0 * y * (0.729325 * std::sqrt(-1-2.88 * (x * x + y * y)+ std::pow((1+(x * x + y * y)),2.88)))/((x * x + y * y) 
               * std::pow((1+(x * x + y * y)),1.44));
  });
  B.set_values(1, [B0](auto x, auto y, auto z) {
    return B0 * x * (0.729325 * std::sqrt(-1-2.88 * (x * x + y * y)+ std::pow((1+(x * x + y * y)),2.88)))/((x * x + y * y) 
              * std::pow((1+(x * x + y * y)),1.44)); 
  });
  B.set_values(2, [B0](auto x, auto y, auto z) { 
        return B0 /std::pow((1+(x * x + y * y)),1.44);
  });

  
  ptc_injector_dynamic<Conf> injector(grid);

  // Jet particles
  injector.inject_pairs( 
      // Injection criterion
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) {return 2 * ppc; },
      [n, kT, B0] LAMBDA(auto &x_global,
                                      rand_state &state, PtcType type) {
        auto x = x_global[0];
        auto y = x_global[1];

        value_t beta_d_x =  x * (2.88 * B0)/(n * std::pow((1+(x * x + y * y)),2.44));

        value_t beta_d_y =  y * (2.88 * B0)/(n * std::pow((1+(x * x + y * y)),2.44));

        value_t beta_d_z =  ((-2.10046 * std::pow((1+(x * x + y * y)),2.44)+std::pow((1+(x * x + y * y)),1.44) 
                              * (2.10046 +6.04931 * (x * x + y * y))) * B0)/(n * std::sqrt(x * x + y * y) 
                                * std::pow((1+(x * x + y * y)),3.44) * std::sqrt(-1.0 -2.88 * (x * x + y * y)
                                  + std::pow((1+(x * x + y * y)),2.88)));

        vec_t<value_t, 3> u = rng_maxwell_juttner_3d(state, kT);

        auto G = 1.0f / math::sqrt(1.0f - (beta_d_x * beta_d_x + beta_d_y * beta_d_y + beta_d_z * beta_d_z));
        auto u0 = math::sqrt(1.0f + u.dot(u));
        auto vdotu = u[1] * beta_d_x + u[2] * beta_d_y + u[0] * beta_d_z;

        auto ran = rng_uniform<value_t>(state);
        if (-beta_d_z * u[0] / u0 > ran) u[0] = -u[0];
        if (-beta_d_x * u[1] / u0 > ran) u[1] = -u[1];
        if (-beta_d_y * u[2] / u0 > ran) u[2] = -u[2];

        u[0] = (1.0/ (1.0 - vdotu)) * ((u[0]/G) - beta_d_z + (G/(G+1)) * vdotu * beta_d_z);
        u[1] = (1.0/ (1.0 - vdotu)) * ((u[1]/G) - beta_d_x + (G/(G+1)) * vdotu * beta_d_x);
        u[2] = (1.0/ (1.0 - vdotu)) * ((u[2]/G) - beta_d_y + (G/(G+1)) * vdotu * beta_d_y);

        value_t sign = 1.0f;
        if (type == PtcType::electron) sign *= -1.0f;

        auto p1 = u[1] * sign;
        auto p2 = u[2] * sign;
        auto p3 = u[0] * sign;
        return vec_t<value_t, 3>(p1, p2, p3);
        },
      // Particle weight
      [n, ppc] LAMBDA(auto &x_global, PtcType type) {
        auto x = x_global[0];
        auto y = x_global[1];
        return n/2.0/ppc;
        });

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}


template void kink_deriven<Config<3>>(vector_field<Config<3>> &B,
                                                     particle_data_t &ptc,
                                                     rng_states_t<exec_tags::device> &states);
                                      

}  // namespace Aperture
