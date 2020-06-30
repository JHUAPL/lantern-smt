#!/usr/bin/env python3

import numpy as np
import torch.nn as nn
from lantern import as_z3


def main():
   
    # Initialize a Pytorch network
    # Lantern currently supports Linear, ReLU, and Hardtanh layers
    net = nn.Sequential(
                nn.Linear(2, 5),
                nn.ReLU(),
                nn.Linear(5, 1),
                nn.ReLU())

    # Normally, we would train this network to compute some function. However
    # since the weights have been automatically initialized, it already
    # computes some function!

    # ... HERE

    constraints, in_vars, out_vars = as_z3(net)

    constraints

    in_vars

    out_vars



if __name__ == "__main__":
    main()
