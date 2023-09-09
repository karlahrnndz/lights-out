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
        self.action_mtx = None
        self.solution = None
        self.solved = False
        self.generated = False

        # Generate state if required
        if not self.state:
            self.state = self.generate_state(dim)
            self.generated = True  # Important because we know generated puzzles have at least one solution

    @staticmethod
    def value_check(state, dim):
        if state:
            state = dok_array(state, dtype=np.int8)

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
        state = dok_array((dim, dim), dtype=np.int8)
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

    @staticmethod
    def prettify(matrix):
        return matrix.toarray().astype(int)

    def create_action_mtx(self):

        # Construct the LHS of the extended matrix (the unravelled action matrix)
        data = np.ones((3, self.no_switches))
        data = np.append(data,
                         [np.tile([1] * (self.dim - 1) + [0], self.no_switches // 3 + 1)[:self.no_switches],
                          np.tile([0, 1, 1], self.no_switches // 3 + 1)[:self.no_switches]],
                         axis=0)
        offsets = np.array([0, -self.dim, self.dim, -1, 1])

        return dok_array(dia_array((data, offsets),
                                   shape=(self.no_switches, self.no_switches),
                                   dtype=np.int8))

    def unravel_state(self, state: Union[ArrayLike, int]):

        if isinstance(state, int):
            state = dok_array(np.array([state for _ in range(self.no_switches)]).reshape(self.no_switches, 1),
                              dtype=np.int8)

        else:
            state = dok_array(np.array(state).reshape(1, self.no_switches), dtype=np.int8)

        return state

    def solve(self, desired_state: Union[ArrayLike, int], parallelize: bool = False):

        # Create action matrix and unravel desired state
        self.action_mtx = self.create_action_mtx()
        self.solution = self.unravel_state(desired_state)

        # Implement gaussian elimination
        self.gauss_elim(parallelize)

    def gauss_elim(self, parallelize: bool = False):

        # Forward elimination
        for k in range(self.no_switches - 1):

            if self.action_mtx[k, k] == 0:  # Check if pivot needed
                piv_on = None
                r = k + 1

                while (not piv_on) and (r <= (self.no_switches - 1)):
                    if self.action_mtx[r, k] == 1:
                        piv_on = r
                    r += 1

                if piv_on:  # Perform pivot of rows k and r, for column k through end (also pivot solution)

                    for c in range(k, self.no_switches):
                        temp = self.action_mtx[k, c]
                        self.action_mtx[k, c] = self.action_mtx[piv_on, c]
                        self.action_mtx[piv_on, c] = temp

                    temp = self.solution[k, 0]
                    self.solution[k, 0] = self.solution[piv_on, 0]
                    self.solution[piv_on, 0] = temp

            if self.action_mtx[k, k] != 0:

                for i in range(k + 1, self.no_switches):

                    for j in range(k + 1, self.no_switches):
                        self.action_mtx[i, j] = self.action_mtx[i, j] ^ self.action_mtx[i, k] * self.action_mtx[k, j]

                    self.solution[i, 0] = self.solution[i, 0] ^ self.action_mtx[i, k] * self.solution[k, 0]
                    self.action_mtx[i, k] = 0  # Variable no longer needed. Zero out to save memory.

        # Backward substitution
        for i in reversed(range(self.no_switches)):
            for j in range(i, self.no_switches):
                self.solution[i, 0] = self.solution[i, 0] ^ self.action_mtx[i, j] * self.solution[j, 0]

            if self.action_mtx[i, i] != 0:
                self.solution[i, 0] = self.solution[i, 0] / self.action_mtx[i, i]

            elif self.solution[i, 0] == 0:  # Solving 0x = 0 which has two solutions in Z2: 0 and 1.
                self.solution[i, 0] = 0

            else:
                raise ValueError("No solution exists.")

        # Change solved flag to True
        self.solved = True


# =================================================================== #
#                              Execution                              #
# =================================================================== #

if __name__ == "__main__":
    my_puzzle = Puzzle(dim=3, seed=SEED)
    print(my_puzzle.prettify(my_puzzle.state))
    my_puzzle.solve(desired_state=1)
