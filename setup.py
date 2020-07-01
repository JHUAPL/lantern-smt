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

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lantern-smt",
    version="0.1.0",
    author="Joshua Brule",
    author_email="joshua.brule@jhuapl.edu",
    license="3-Clause BSD License",
    description="Tools to encode PyTorch modules as Z3 constraints",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JHUAPL/lantern-smt",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3",
    install_requires=["torch",
                      "z3-solver"]
)
