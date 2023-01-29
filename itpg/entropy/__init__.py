"""Entropy functions and utils."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# License: MIT License

from .permut_entropy import (permutation_entropy, permutation_entropy_map,
                             ordinal_patterns)

from .entropy_gaussian import (entropy_gauss, entropy_gauss_loop,
                               entropy_gauss_nd)
