import numpy as np
from numpy.typing import ArrayLike
from numpy.random import default_rng
from scipy.sparse import dia_array
from typing import Union
import warnings
import time


# =================================================================== #
#                              Constants                              #
# =================================================================== #

SEED = 42


# =================================================================== #
#                               Classes                               #
# =================================================================== #

class Puzzle:
    """A representation of the lights-off start_state's start_state."""

    def __init__(self,
                 start_state: ArrayLike = None,
                 gen_state: tuple = None,
                 seed: int = None):

        # Initial value check and config
        self.start_state, self.gen_state, self.transposed = self.value_check(start_state, gen_state)
        if self.start_state is None:
            self.gen = default_rng(seed=seed)
            self.start_state = self.generate_state()

        self.dim = self.start_state.shape
        self.max_dim = max(self.dim[0], self.dim[1])
        self.no_switches = self.dim[0] * self.dim[1]
        self.action_mtx = None
        self.solution = None

    @staticmethod
    def value_check(start_state, gen_state):
        transposed = False

        if start_state is not None:
            start_state = np.array(start_state, dtype=np.int8)

            if start_state.shape[0] > start_state.shape[1]:
                start_state = start_state.T
                transposed = True

            # Warn when gen_state is present as well as staten and force gen_state to match start_state
            if gen_state:
                warnings.warn(f"Both `start_state` and `gen_state` were provided."
                              f"Setting `gen_state = None.")
                gen_state = None

            # dim = (start_state.shape[0], start_state.shape[1])

        elif gen_state is None:
            raise ValueError("One of `start_state` and `gen_state` must be provided.")

        elif gen_state[0] > gen_state[1]:
            gen_state = (gen_state[1], gen_state[0])
            transposed = True

        return start_state, gen_state, transposed

    def generate_state(self):
        state = np.zeros(self.gen_state, dtype=np.int8)
        for i in range(self.gen_state[0]):
            for j in range(self.gen_state[1]):

                press_switch = self.gen.binomial(1, 0.5, 1)[0]
                if press_switch:
                    affected = {(i, j),
                                (i, min(j + 1, self.gen_state[1] - 1)),
                                (i, max(j - 1, 0)),
                                (min(i + 1, self.gen_state[0] - 1), j),
                                (max(i - 1, 0), j)}

                    for button in affected:
                        state[button] = state[button] ^ True  # XOR with True to flip switch

        return state

    def create_action_mtx(self):

        if self.dim == (1, 1):
            return np.array([[1]], dtype=np.int8)

        else:  # Construct the LHS of the extended matrix (the unravelled action matrix)
            data = np.ones((3, self.max_dim ** 2))
            data = np.append(data, [
                np.tile([1] * (self.max_dim - 1) + [0], (self.max_dim ** 2) // 3 + 1)[:self.max_dim ** 2],
                np.tile([0] + [1] * (self.max_dim - 1), (self.max_dim ** 2) // 3 + 1)[:self.max_dim ** 2]], axis=0)
            offsets = np.array([0, -self.max_dim, self.max_dim, -1, 1])
            action_mtx = (dia_array((data, offsets),
                                    shape=(self.max_dim ** 2, self.max_dim ** 2),
                                    dtype=np.int8)).toarray()

        if self.no_switches != self.max_dim ** 2:  # Need to correct action matrix (always more columns than rows)
            sel_idx = [ele for ele in range(self.no_switches)]
            action_mtx = action_mtx[sel_idx, :][:, sel_idx]

        return action_mtx

    def prepare_end_state(self, desired_state: Union[ArrayLike, int]):

        if isinstance(desired_state, int):
            desired_state = np.array([desired_state for _ in range(self.no_switches)], dtype=np.int8)

        else:
            desired_state = np.array(desired_state, dtype=np.int8).ravel()

        return np.mod(self.start_state.ravel() + desired_state, 2)

    def solve(self, end_state: Union[ArrayLike, int], parallelize: bool = False):  # TODO - Enable parallelization

        # Create action matrix and unravel desired start_state
        self.action_mtx = self.create_action_mtx()
        self.solution = self.prepare_end_state(end_state)

        # Implement gaussian elimination
        self.gauss_elim(parallelize)

    def gauss_elim(self, parallelize: bool = False):

        row_map = {i: i for i in range(self.no_switches)}

        # Forward elimination
        for k in range(self.no_switches - 1):
            k_idx = row_map[k]

            if self.action_mtx[k_idx, k] == 0:  # Check if pivot needed
                piv_on = None
                r = k + 1

                while (not piv_on) and (r <= (self.no_switches - 1)):
                    r_idx = row_map[r]
                    if self.action_mtx[r_idx, k] == 1:
                        piv_on = r
                    r += 1

                if piv_on:  # Perform pivot of rows k and r, for column k through end (also pivot solution)
                    temp = row_map[k]
                    row_map[k] = row_map[piv_on]
                    row_map[piv_on] = temp
                    k_idx = row_map[k]

            if self.action_mtx[k_idx, k] != 0:
                for i in range(k + 1, self.no_switches):
                    i_idx = row_map[i]
                    coeff = self.action_mtx[i_idx, k]
                    self.action_mtx[i_idx, :] = np.mod(self.action_mtx[i_idx, :] + coeff * self.action_mtx[k_idx, :], 2)
                    self.solution[i_idx] = (self.solution[i_idx] + coeff * self.solution[k_idx]) % 2

        # Backward substitution
        for i in reversed(range(self.no_switches)):
            i_idx = row_map[i]

            if i != (self.no_switches - 1):
                for j in range(i + 1, self.no_switches):
                    j_idx = row_map[j]
                    self.solution[i_idx] = \
                        (self.solution[i_idx] + self.action_mtx[i_idx, j] * self.solution[j_idx]) % 2

            if self.action_mtx[i_idx, i] != 0:
                self.solution[i_idx] = (self.solution[i_idx] / self.action_mtx[i_idx, i]) % 2

            elif self.solution[i_idx] == 0:  # Solving 0x = 0 which has two solutions in Z2: 0 and 1.
                self.solution[i_idx] = 0

            else:
                raise ValueError("No solution exists.")

        # Format solution and set solved flag
        self.solution = np.array([self.solution[row_map[i]] for i in range(self.no_switches)]).reshape(self.dim)
        if self.transposed:
            self.solution = self.solution.T


# =================================================================== #
#                              Execution                              #
# =================================================================== #

if __name__ == "__main__":
    start = time.time()
    puzzle = Puzzle(start_state=np.zeros((3, 2)), gen_state=None, seed=SEED)
    puzzle.solve(end_state=1)
    end = time.time()
    print(puzzle.solution)
    print(end - start)
