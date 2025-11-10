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

//__device__ int print_counter = 0;

template <typename Conf>
void
kink_Force_Free(vector_field<Conf> &B, particle_data_t &ptc,
                       rng_states_t<exec_tags::device> &states) {
  using value_t = typename Conf::value_t;
  value_t B0 = sim_env().params().get_as<double>("B0", 200.0);
  value_t pitch = sim_env().params().get_as<double>("pitch_parameter", 1.44);
  //value_t sigma = sim_env().params().get_as<double>("sigma", 10.0);
  value_t kT = sim_env().params().get_as<double>("Temperature", 0.01);
  value_t n = sim_env().params().get_as<double>("number_density", 360.0);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);
  value_t ppc = sim_env().params().get_as<int64_t>("ppc", 4);
  

  auto &grid = B.grid();
  auto ext = grid.extent();

 
  // Initialize the magnetic field values
  auto Bz = [B0, pitch] (auto r) {
    return B0 / std::pow((1 + square(r)), 1.44);
  };

  auto Bphi = [B0, pitch] (auto r) {
    return B0 * sqrt((std::pow((1 + square(r)), 2.0 * pitch) - 2.0 * pitch * square(r) - 1.0)
                /(2.0 * pitch - 1.0)) / (r * std::pow((1 + square(r)), pitch)) ;
  };



  // Initialize the magnetic field values
  B.set_values(0, [Bphi](auto x, auto y, auto z) {
    auto r = math::sqrt(x*x + y*y);
    return -y * Bphi(r) / r;
  });
  B.set_values(1, [Bphi](auto x, auto y, auto z) {
    auto r = math::sqrt(x*x + y*y);
    return x * Bphi(r) / r;
  });
  B.set_values(2, [Bz](auto x, auto y, auto z) { 
    auto r = math::sqrt(x*x + y*y);
    return Bz(r); 
  });

  // profiles of physical quantities
  auto jz_profile = [B0, pitch] LAMBDA (auto r) {
    if (r < 10e-2) {
        // Return the value of jz_profile for r = 10^-2
        return B0 * 2.0 * pitch * square(10e-2)  
               / sqrt((std::pow((1 + square(10e-2)), 2.0 * pitch) - 2.0 * pitch * square(10e-2) - 1.0) / (2.0 * pitch - 1.0))
               / std::pow((1 + square(10e-2)), 1.0 + pitch);
    } else {
        // Return the normal jz_profile value for the given r
        return B0 * 2.0 * pitch * square(r)  
               / sqrt((std::pow((1 + square(r)), 2.0 * pitch) - 2.0 * pitch * square(r) - 1.0) / (2.0 * pitch - 1.0))
               / std::pow((1 + square(r)), 1.0 + pitch);
    }
  };

  auto jphi_profile = [B0, pitch] LAMBDA (auto r) {
    return B0 * 2.0 * pitch * r  / std::pow((1 + square(r)), 1.0 + pitch)  ;
  };

  
  ptc_injector_dynamic<Conf> injector(grid);

  // Jet particles
  injector.inject_pairs( 
      // Injection criterion
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) {return 2 * ppc; },
      [n, jphi_profile, jz_profile, kT] LAMBDA(auto &x_global,
                                      rand_state &state, PtcType type) {
        auto x = x_global[0];
        auto y = x_global[1];
        auto r = math::sqrt(x*x + y*y);

        value_t beta_d_x = - x * jphi_profile(r)/ (n * 2.0 * r) ;

        value_t beta_d_y =  y * jphi_profile(r)/ (n * 2.0 * r) ;

        value_t beta_d_z =  jz_profile(r)/ (n * 2.0) ;

        //printf("beta_d_x: %f, beta_d_y: %f, beta_d_z: %f\n", beta_d_x, beta_d_y, beta_d_z);

        vec_t<value_t, 3> u = rng_maxwell_juttner_3d(state, kT);

        auto G = 1.0f / math::sqrt(1.0f - (beta_d_x * beta_d_x + beta_d_y * beta_d_y + beta_d_z * beta_d_z));
        auto u0 = math::sqrt(1.0f + u.dot(u));
        //auto vdotu = u[1] * beta_d_x + u[2] * beta_d_y + u[0] * beta_d_z;

        
        //if (atomicAdd(&print_counter, 1) < 1000) {
        //    printf("Particle %d -> beta_d_x: %f, beta_d_y: %f, beta_d_z: %f,  G: %f\n", 
        //            print_counter, beta_d_x, beta_d_y, beta_d_z, G);
        //  }


        auto ran = rng_uniform<value_t>(state);
        if (-beta_d_z * u[0] / u0 > ran) u[0] = -u[0];
        if (-beta_d_x * u[1] / u0 > ran) u[1] = -u[1];
        if (-beta_d_y * u[2] / u0 > ran) u[2] = -u[2];
        
        u[0] = u[0] + beta_d_z * G ;
        u[1] = u[1] + beta_d_x * G ;
        u[2] = u[2] + beta_d_y * G ;


        value_t sign = 1.0f;
        if (type == PtcType::electron) sign *= -1.0f;

        auto p1 = u[1] * sign;
        auto p2 = u[2] * sign;
        auto p3 = u[0] * sign;
        return vec_t<value_t, 3>(p1, p2, p3);
        },
      // Particle weight
      [n, ppc, q_e] LAMBDA(auto &x_global, PtcType type) {
        return n/q_e/ppc;
        });

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}



template void kink_Force_Free<Config<3>>(vector_field<Config<3>> &B,
                                                particle_data_t &ptc,
                                                rng_states_t<exec_tags::device> &states);
                                      

}  // namespace Aperture
