"""
Lantern: safer than a torch

The Lantern package contains utility funcitons to support formal
verification of PyTorch modules by encoding the behavior of (certain)
neural networks as Z3 constraints.

The 'public' API includes:

- round_model(model, sbits)
- as_z3(model, sort, prefix)
"""

# Copyright 2020 The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
#
# Licensed under the 3-Clause BSD License (the "License");
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

import copy
import struct
from collections import OrderedDict
from functools import reduce

import torch.nn as nn
import z3


def truncate_double(f, sbits=52):
    """
    Truncate the significand/mantissa precision of f to number of sbits.

    Note that f is expected to be a Python float (double precision).

    sbits=52 is a no-op
    """
    assert((sbits <= 52) and (sbits >= 0))

    original = float(f)
    int_cast = struct.unpack(">q", struct.pack(">d", original))[0]
    truncated_int = ((int_cast >> (52 - sbits)) << (52 - sbits))
    truncated = float(struct.unpack(">d", struct.pack(">q", truncated_int))[0])

    return truncated


def round_model(model, sbits=52):
    """
    Return a new model where every value in the original state dict has
    had its fractional precision reduced to number of sbits. Exponent
    part remains the same (11 bits) so the result can be returned as
    a Python float.

    Note that sbits=52 is a no-op. Single precision sbits=23; half=10
    """
    new_model = copy.deepcopy(model)

    for t in new_model.state_dict().values():
        t.apply_(lambda f: truncate_double(f, sbits))

    return new_model


def encode_relu(x, y):
    """
    Returns a list of z3 constraints corresponding to:

    y == relu(x)

    Where: x, y are lists of z3 variables
    """
    assert len(x) == len(y)

    constraints = []
    for x_i, y_i in zip(x, y):
        lhs = y_i
        rhs = z3.If(x_i >= 0, x_i, 0)
        constraint = z3.simplify(lhs == rhs)
        constraints.append(constraint)

    return constraints


def encode_hardtanh(x, y, min_val=-1, max_val=1):
    """
    Returns a list of z3 constraints corresponding to:

    y == hardtanh(x, min_val=-1, max_val=1)

    Where: x, y are lists of z3 variables
    """
    assert len(x) == len(y)
    assert min_val < max_val

    constraints = []
    for x_i, y_i in zip(x, y):
        lhs = y_i
        rhs = z3.If(x_i <= min_val,
                    min_val,
                    z3.If(x_i <= max_val,
                          x_i,
                          max_val))
        constraint = z3.simplify(lhs == rhs)
        constraints.append(constraint)

    return constraints


def hacky_sum(coll):
    """
    Because z3.Sum() doesn't work on FP sorts
    """
    if len(coll) == 0:
        return 0
    elif len(coll) == 1:
        return coll[0]
    else:
        return reduce(lambda x, y: x + y, coll)


def encode_linear(W, b, x, y):
    """
    Returns a list of z3 constraints corresponding to:

    y == W * x + b

    Where: x, y are lists of z3 variables,
           W, b are pytorch tensors
    """
    m, n = W.size()
    assert m == len(b)
    assert n == len(x)
    assert m == len(y)
    assert m >= 1 and n >= 1

    constraints = []
    for i in range(m):
        lhs = y[i]
        rhs = hacky_sum([W[i, j].item() * x[j] for j in range(n)]) + b[i].item()
        constraint = z3.simplify(lhs == rhs)
        constraints.append(constraint)

    return constraints


def const_vector(prefix, length, sort=z3.RealSort()):
    """
    Returns a list of z3 constants of given sort.

    e.g. const_vector("foo", 5, z3.FloatSingle())
    Returns a list of 5 FP
    """
    names = [prefix + "__" + str(i) for i in range(length)]
    return z3.Consts(names, sort)


def as_z3(model, sort=z3.RealSort(), prefix=""):
    """
    Calculate z3 constraints from a torch.nn.Sequential model.

    Returns (constraints, z3_input, z3_output) where:

    - constraints is a list of z3 constraints for the entire network
    - z3_input is z3.RealVector representing the input to the network
    - z3_output is a z3.RealVector representing output of the network

    There are several caveats:

    - The model must be a torch Sequential
    - The first layer must be Linear
    - Dropout layers are ignored
    - Identity layers are ignored
    - Supported layers are: Linear, ReLU, Hardtanh, Dropout, Identity
    - An Exception is raised on any other type of layer

    sort defaults to z3.RealSort(), but floating point sorts are
    permitted; note that z3.FloatSingle() matches the default behavior
    of PyTorch more accurately (but has different performance
    characteristics compared to a real arithmetic theory

    prefix is an optional string prefix for the generated z3 variables
    """
    assert isinstance(model, nn.Sequential)

    modules = OrderedDict(model.named_modules())

    # named_modules() has ("" -> the entire net) as first key/val pair; remove
    modules.pop("")

    constraints = []
    first_vector = None
    previous_vector = None
    for name in modules:
        module = modules[name]

        if isinstance(module, nn.Linear):
            W, b = module.parameters()

            in_vector = previous_vector
            if in_vector is None:
                in_vector = const_vector("{}_lin{}_in".format(prefix, name),
                                         module.in_features, sort)
                first_vector = in_vector

            out_vector = const_vector("{}_lin{}_out".format(prefix, name),
                                      module.out_features, sort)

            constraints.extend(encode_linear(W, b, in_vector, out_vector))

        elif isinstance(module, nn.ReLU):
            in_vector = previous_vector
            if in_vector is None:
                raise ValueError("First layer must be linear")

            out_vector = const_vector("{}_relu{}_out".format(prefix, name),
                                      len(in_vector), sort)

            constraints.extend(encode_relu(in_vector, out_vector))

        elif isinstance(module, nn.Hardtanh):
            in_vector = previous_vector
            if in_vector is None:
                raise ValueError("First layer must be linear")

            out_vector = const_vector("{}_tanh{}_out".format(prefix, name),
                                      len(in_vector), sort)

            constraints.extend(encode_hardtanh(in_vector, out_vector,
                                               module.min_val, module.max_val))

        elif isinstance(module, nn.Dropout):
            pass
        elif isinstance(module, nn.Identity):
            pass
        else:
            raise ValueError("Don't know how to convert module: {}".format(module))

        previous_vector = out_vector


    # previous_vector is vector associated with last layer output
    return (constraints, first_vector, previous_vector)

