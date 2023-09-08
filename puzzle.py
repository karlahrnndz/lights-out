import numpy as np
from numpy.typing import ArrayLike
from numpy.random import default_rng
from scipy.sparse import dok_array, dia_array
from typing import Union
import warnings

# =================================================================== #
#                              Constants                              #
# =================================================================== #

SEED = 42


# =================================================================== #
#                               Classes                               #
# =================================================================== #

class Puzzle:
    """A representation of the lights-off state's state."""

    def __init__(self,
                 state: ArrayLike = None,
                 dim: int = None,
                 seed: int = None):

        # Initial value check and config
        self.state, self.dim = self.value_check(state, dim)
        self.gen = default_rng(seed=seed)
        self.no_switches = self.dim ** 2
        self.desired_state = None
        self.action_mtx = None
        self.solution = None

        # Generate state if required
        if not self.state:
            self.state = self.generate_state(dim)

    @staticmethod
    def value_check(state, dim):
        if state:
            state = dok_array(state, dtype=np.bool_)

            # Make sure state has the correct dimensions
            if state.shape[0] != state.shape[1]:
                raise ValueError("`state` should be a 2D Python array_like object "
                                 "with the same number or rows as columns.")

            # Warn when dim is present as well as staten and force dim to match state
            if dim:
                warnings.warn(f"Both `state` and `dim` were provided."
                              f"Setting `dim = state.shape[0]`.")

            dim = state.shape[0]

        elif not dim:
            raise ValueError("One of `state` and `dim` must be provided.")

        return state, dim

    def generate_state(self, dim):
        state = dok_array((dim, dim), dtype=np.bool_)
        for i in range(dim):
            for j in range(dim):

                press_switch = self.gen.binomial(1, 0.5, 1)[0]
                if press_switch:
                    affected = {(i, j),
                                (i, min(j + 1, dim - 1)),
                                (i, max(j - 1, 0)),
                                (min(i + 1, dim - 1), j),
                                (max(i - 1, 0), j)}

                    for button in affected:
                        state[button] = state[button] ^ True  # XOR with True to flip switch

        return state

    def prettify_state(self):
        return self.state.toarray().astype(int)

    def create_action_mtx(self):

        # Construct the LHS of the extended matrix (the unravelled action matrix)
        data = np.ones((3, self.no_switches))
        data = np.append(data,
                         [np.tile([1, 1, 0], self.no_switches // 3 + 1)[:self.no_switches],
                          np.tile([0, 1, 1], self.no_switches // 3 + 1)[:self.no_switches]],
                         axis=0)
        offsets = np.array([0, -3, 3, -1, 1])

        return dok_array(dia_array((data, offsets),
                                   shape=(self.no_switches, self.no_switches),
                                   dtype=np.bool_))

    def unravel_state(self, state: Union[ArrayLike, bool]):

        if isinstance(state, bool):
            state = dok_array(np.array([[state] for _ in range(self.no_switches)]))

        else:
            state = dok_array(np.array(state).reshape(self.no_switches, 1), dtype=np.bool_)

        return state

    def solve_puzzle(self, desired_state: Union[ArrayLike, bool], parallelize: bool = False):

        # Create action matrix and unravel desired state
        self.action_mtx = self.create_action_mtx()
        self.desired_state = self.unravel_state(desired_state)

        # Implement gaussian elimination
        self.solution = self.gauss_elim(parallelize)

    def gauss_elim(self, parallelize: bool = False):

        # Implement forward elimination

        # Implement backward substitution

        # Format solution

        return self.solution


# =================================================================== #
#                              Execution                              #
# =================================================================== #

if __name__ == "__main__":
    my_puzzle = Puzzle(dim=5, seed=SEED)
    print(my_puzzle.prettify_state())
