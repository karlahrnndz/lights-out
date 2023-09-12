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
    """A representation of the lights-off state's state."""

    def __init__(self,
                 state: ArrayLike = None,
                 dim: Union[int, tuple] = None,
                 seed: int = None):

        # Initial value check and config
        self.state, self.dim = self.value_check(state, dim)  # TODO - Add solution between any states
        self.gen = default_rng(seed=seed)
        self.no_switches = self.dim[0] * self.dim[1]
        self.action_mtx = None
        self.solution = None
        self.solved = False
        self.max_dim = max(self.dim[0], self.dim[1])

        # Generate state if required
        if not self.state:
            self.state = self.generate_state()

    @staticmethod
    def value_check(state, dim):
        if state:
            state = np.array(state, dtype=np.int8)

            if state.shape[0] > state.shape[1]:
                warnings.warn(f"`state.shape[0] > state.shape[1]` while this code assumes "
                              f"`state.shape[0] < state.shape[1]`. Will transpose the problem.")
                state = state.T

            # Warn when dim is present as well as staten and force dim to match state
            if dim:
                warnings.warn(f"Both `state` and `dim` were provided."
                              f"Setting `dim = (state.shape[0], state.shape[1])`.")

            dim = (state.shape[0], state.shape[1])

        elif not dim:
            raise ValueError("One of `state` and `dim` must be provided.")

        elif isinstance(dim, int):
            dim = (dim, dim)

        elif dim[0] > dim[1]:
            warnings.warn(f"`dim[0] > dim[1]` while this code assumes `dim[0] <= dim[1]`. "
                          f"Will transpose the problem so that `dim[0] < dim[1]`.")
            dim = (dim[1], dim[0])

        return state, dim

    def generate_state(self):

        state = np.zeros(self.dim, dtype=np.int8)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):

                press_switch = self.gen.binomial(1, 0.5, 1)[0]
                if press_switch:
                    affected = {(i, j),
                                (i, min(j + 1, self.dim[1] - 1)),
                                (i, max(j - 1, 0)),
                                (min(i + 1, self.dim[0] - 1), j),
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
            action_mtx = np.array(action_mtx)[sel_idx, sel_idx]

        return action_mtx

    def unravel_state(self, state: Union[ArrayLike, int]):

        if isinstance(state, int):
            state = np.array([state for _ in range(self.no_switches)])

        else:
            state = np.array(state).ravel()

        return state

    def solve(self, desired_state: Union[ArrayLike, int], parallelize: bool = False):  # TODO - Enable parallelization

        # Create action matrix and unravel desired state
        self.action_mtx = self.create_action_mtx()
        self.solution = self.unravel_state(desired_state)

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

            elif self.solution[i_idx, 0] == 0:  # Solving 0x = 0 which has two solutions in Z2: 0 and 1.
                self.solution[i_idx, 0] = 0

            else:
                raise ValueError("No solution exists.")

        # Format solution and set solved flag
        self.solution = np.array([self.solution[row_map[i]] for i in range(self.no_switches)]).reshape(self.dim)
        self.solved = True


# =================================================================== #
#                              Execution                              #
# =================================================================== #

if __name__ == "__main__":
    start = time.time()
    my_puzzle = Puzzle(dim=20, seed=SEED)
    my_puzzle.solve(desired_state=1)
    end = time.time()
    print(end - start)
    print(my_puzzle.solution)
