from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import unittest
import pyMKL

class TestMKLUtils(unittest.TestCase):

    def test_mkl_get_version(self):
        versionString = pyMKL.mkl_get_version()
        print(versionString)

    def test_mkl_get_max_threads(self):
        max_threads = pyMKL.mkl_get_max_threads()
        print("Max. Threads: {}".format(max_threads))

    def test_mkl_set_num_threads(self):
        pyMKL.mkl_set_num_threads(1)
        pyMKL.mkl_set_num_threads(pyMKL.mkl_get_max_threads())

if __name__ == '__main__':
    unittest.main()
