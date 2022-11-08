"""
The code for the pyMKL module is taken and modified from the work of David Marchant.
From: https://github.com/dwfmarchant/pyMKL
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
# from future import standard_library
# standard_library.install_aliases()

import platform
import numpy as np
import scipy.sparse as sp
from .loadMKL import _loadMKL

MKLlib = _loadMKL()

from .MKLutils import mkl_get_version, mkl_get_max_threads, mkl_set_num_threads
from .pardisoInterface import pardisoinit, pardiso
from .pardisoSolver import pardisoSolver
