/*
 * Copyright (c) 2022 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _SYNC_CURV_EMISSION_H_
#define _SYNC_CURV_EMISSION_H_

#include "core/multi_array.hpp"
#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/physics/sync_emission_helper.hpp"

namespace Aperture {

class sync_curv_emission_t_F : public system_t {
 public:
  using value_t = Scalar;
  static std::string name() { return "sync_curv_emission_F"; }

  sync_curv_emission_t_F(MemType type = MemType::host_device);
  ~sync_curv_emission_t_F();

  void compute_lookup_table_F();

  sync_emission_helper_t get_helper() {
    return m_sync;
  }

 private:
  int m_nx;
  value_t m_x_min, m_x_max;
  sync_emission_helper_t m_sync;
  buffer<value_t> m_Fx_lookup, m_Fx_cumulative;
};

class sync_curv_emission_t_G : public system_t {
 public:
  using value_t = Scalar;
  static std::string name() { return "sync_curv_emission_G"; }

  sync_curv_emission_t_G(MemType type = MemType::host_device);
  ~sync_curv_emission_t_G();

  void compute_lookup_table_G();

  sync_emission_helper_t get_helper() {
    return m_sync;
  }

 private:
  int m_nx;
  value_t m_x_min, m_x_max;
  sync_emission_helper_t m_sync;
  buffer<value_t> m_Gx_lookup, m_Gx_cumulative;
};

}

#endif  // _SYNC_CURV_EMISSION_H_
