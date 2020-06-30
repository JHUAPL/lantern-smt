# Copyright 2020 The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
#
# Licensed under the 3-Caluse BSD License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torch
import z3

import lantern

class TestEncode(unittest.TestCase):

    def test_hardtanh(self):
        """
        This test is a mild sanity check; don't take too seriously
        """
        y = z3.RealVector("y", 2)
        x = z3.RealVector("x", 2)
        constraints = lantern.encode_hardtanh(x, y)

        s = z3.SolverFor("QF_LRA")

        s.add(constraints)

        # x_0 = -2 => y_0 = -1
        s.add(x[0] == -2)
        s.add(x[1] == y[0])
        self.assertTrue(s.check() == z3.sat)

        # should fail, since y_1 is constrained to -1
        s.add(y[1] == 0)
        self.assertTrue(s.check() == z3.unsat)

