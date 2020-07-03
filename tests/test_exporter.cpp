/*
 * Copyright (c) 2020 Alex Chen.
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

#include "catch.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/data_exporter.h"

using namespace Aperture;

TEST_CASE("Writing a grid to file", "[data_output]") {
  sim_environment env;

  // SECTION("1D grid") {
  //   typedef Config<1> Conf;
  //   env.params().add("N", std::vector<int64_t>({32}));
  //   env.params().add("guard", std::vector<int64_t>({2}));
  //   env.params().add("size", std::vector<double>({1.0}));
  //   env.params().add("lower", std::vector<double>({0.0}));

  //   grid_t<Conf> grid(env);
  //   data_exporter<Conf> exporter(env, grid);

  //   exporter.init();
  //   exporter.write_grid();
  // }

  SECTION("2D grid") {
    typedef Config<2> Conf;
    env.params().add("N", std::vector<int64_t>({40, 40}));
    env.params().add("guard", std::vector<int64_t>({2, 2}));
    env.params().add("size", std::vector<double>({1.0, 2.0}));
    env.params().add("lower", std::vector<double>({0.0, 0.0}));
    env.params().add<int64_t>("downsample", 4);

    grid_t<Conf> grid(env);
    data_exporter<Conf> exporter(env, grid);

    exporter.init();
    exporter.write_grid();

    H5File f("Data/grid.h5");
    auto x1 = f.read_multi_array<float, 2>("x1");
    auto x2 = f.read_multi_array<float, 2>("x2");
    for (auto idx : x1.indices()) {
      auto pos = idx.get_pos();
      REQUIRE(x1[idx] == Approx(grid.template pos<0>(pos[0] * 4 + 2, false)));
      REQUIRE(x2[idx] == Approx(grid.template pos<1>(pos[1] * 4 + 2, false)));
    }
  }
}

// TEST_CASE("Writing a 2D grid", "[data_output]") {
//   sim_environment env;
// }
