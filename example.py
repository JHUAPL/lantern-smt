#!/usr/bin/env python3

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

import torch.nn as nn
import lantern
import z3

def main():
    """Lantern demo"""

    # Initialize a PyTorch network
    # Lantern currently supports: Linear, ReLU, Hardtanh, Dropout, Identity
    net = nn.Sequential(
                nn.Linear(2, 5),
                nn.ReLU(),
                nn.Linear(5, 1),
                nn.ReLU())

    print("A PyTorch network:")
    print(net)
    print()

    # Normally, we would train this network to compute some function. However,
    # for this demo, we'll just use the initialized weights.
    print("Network parameters:")
    print(list(net.parameters()))
    print()

    # lantern.as_z3(model) returns a triple of z3 constraints, input variables,
    # and output variables that directly correspond to the behavior of the 
    # given PyTorch network. By default, latnern assumes Real-sorted variables.
    constraints, in_vars, out_vars = lantern.as_z3(net)

    print("Z3 constraints, input variables, output variables (Real-sorted):")
    print(constraints)
    print(in_vars)
    print(out_vars)
    print()

    # The 'payoff' is that we can prove theorems about our network with z3.
    # Trivially, we can ask for a satisfying assignment of variables
    print("A satisfying assignment to the variables in this network:")
    z3.solve(constraints)
    print()

    # However, we can run the network "backwards"; e.g. what is an *input* that
    # causes the network to output the value 0 (if such an input exists)?
    constraints.append(out_vars[0] == 0)
    print("An assignment such that the output variable is 0:")
    z3.solve(constraints)
    print()
    
    # To more precisely represent the underlying computations, consider using
    # an appropriate floating-point sort; PyTorch defaults to single precision.
    # To speed up satisfiability computations, models can be 'rounded', which
    # truncates the mantissa of every PyTorch model parameter. Note that the
    # exponent part remains the same (11 bits) so that the result can be
    # returned as a Python float. Here, we truncate to 10 bits (half precision).
    rounded_net = lantern.round_model(net, 10)
    constraints, in_vars, out_vars = lantern.as_z3(rounded_net, sort=z3.FPSort(11, 10))
    print("Z3 constraints, input variables, output variables (FPSort(11, 10)):")
    print(constraints)
    print(in_vars)
    print(out_vars)
    print()

    # The constraints and variables are ordinary Z3Py objects and can be
    # composed with additional constraints.
    print("Happy hacking!")


if __name__ == "__main__":
    main()
