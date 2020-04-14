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

