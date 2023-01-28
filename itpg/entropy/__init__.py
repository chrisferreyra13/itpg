"""Entropy functions and utils."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# License: MIT License

from .permut_entropy import (permutation_entropy, permutation_entropy_map,
                             ordinal_patterns)

from .entropy_gaussian import (entropy_g, entropy_g_loop, entropy_g_tensor)
