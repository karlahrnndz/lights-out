"""
Implemented in Python 3.9.
"""

from numpy.typing import ArrayLike
from scipy.sparse import dia_array
from typing import Union
from numba import jit
import numpy as np
import time
import csv


# =================================================================== #
#                               Classes                               #
# =================================================================== #

class Puzzle:
    """A representation of the lights-off puzzle's initial state, solution, and dimensions."""

    def __init__(self,
                 init_state: ArrayLike = None):

        self.init_state, self.transposed = self.standardize_input(init_state)
        self.dim = self.init_state.shape
        self.max_dim = max(self.dim[0], self.dim[1])
        self.no_switches = self.dim[0] * self.dim[1]
        self.toggle_mtx = None
        self.solution = None

    @staticmethod
    def standardize_input(init_state):
        """Make sure inputs adhere to specific object and data types."""
        transposed = False

        if init_state is not None:
            init_state = init_state.astype(np.int8) if isinstance(init_state, np.ndarray) \
                else np.array(init_state, dtype=np.int8)

            if init_state.shape[0] > init_state.shape[1]:
                init_state = init_state.T
                transposed = True

        return init_state, transposed

    def update_final_state(self, final_state: Union[ArrayLike, int]):
        """Combine the initial state and final state into the RHS of the system of linear equations."""

        if isinstance(final_state, int):
            final_state = np.array([final_state for _ in range(self.no_switches)], dtype=np.int8)

        else:
            final_state = np.array(final_state, dtype=np.int8).ravel()

        return np.mod(self.init_state.ravel() + final_state, 2)

    def solve(self, final_state: Union[ArrayLike, int]):
        """Solve the puzzle."""

        # Create toggle matrix and unravel desired init_state
        self.toggle_mtx = self.create_toggle_mtx()
        self.solution = self.update_final_state(final_state)

        # Implement gaussian elimination (function implemented separately to allow use of numba)
        self.solution, self.init_state = \
            gauss_elim(self.no_switches, self.toggle_mtx, self.solution, self.dim, self.transposed, self.init_state)

    def create_toggle_mtx(self):
        """Create the toggle matrix for the grid."""

        if self.dim == (1, 1):
            return np.array([[1]], dtype=np.int8)

        else:  # Construct the LHS of the extended matrix (the unravelled toggle matrix)
            data = np.ones((3, self.max_dim ** 2))
            data = np.append(data, [
                np.tile([1] * (self.max_dim - 1) + [0], (self.max_dim ** 2) // 3 + 1)[:self.max_dim ** 2],
                np.tile([0] + [1] * (self.max_dim - 1), (self.max_dim ** 2) // 3 + 1)[:self.max_dim ** 2]], axis=0)
            offsets = np.array([0, -self.max_dim, self.max_dim, -1, 1])
            toggle_mtx = (dia_array((data, offsets),
                                    shape=(self.max_dim ** 2, self.max_dim ** 2),
                                    dtype=np.int8)).toarray()

        if self.no_switches != self.max_dim ** 2:  # Need to correct toggle matrix (always more columns than rows)
            sel_idx = [ele for ele in range(self.no_switches)]
            toggle_mtx = toggle_mtx[sel_idx, :][:, sel_idx]

        return toggle_mtx


# =================================================================== #
#                              Functions                              #
# =================================================================== #

@jit(nopython=True)
def gauss_elim(no_switches, toggle_mtx, solution, dim, transposed, init_state):
    """Implement gaussian elimination."""
    row_map = list(range(no_switches))

    # Forward elimination
    for k in range(no_switches - 1):
        k_idx = row_map[k]

        if toggle_mtx[k_idx, k] == 0:  # Check if pivot needed
            piv_on = -1
            r = k + 1

            while (piv_on < 0) and (r <= (no_switches - 1)):
                r_idx = row_map[r]

                if toggle_mtx[r_idx, k] == 1:
                    piv_on = r

                r += 1

            if piv_on >= 0:  # Perform pivot of rows k and r, for column k through end (also pivot solution)
                temp = row_map[k]
                row_map[k] = row_map[piv_on]
                row_map[piv_on] = temp
                k_idx = row_map[k]

        if toggle_mtx[k_idx, k] != 0:

            for i in range(k + 1, no_switches):
                i_idx = row_map[i]
                coeff = toggle_mtx[i_idx, k]

                if coeff != 0:
                    toggle_mtx[i_idx, :] = np.mod(toggle_mtx[i_idx, :] + toggle_mtx[k_idx, :], 2)
                    solution[i_idx] = (solution[i_idx] + solution[k_idx]) % 2

    # Backward substitution
    for i in range(no_switches - 1, -1, -1):
        i_idx = row_map[i]

        if i != (no_switches - 1):

            for j in range(i + 1, no_switches):
                j_idx = row_map[j]
                solution[i_idx] = (solution[i_idx] + toggle_mtx[i_idx, j] * solution[j_idx]) % 2

        if toggle_mtx[i_idx, i] != 0:
            solution[i_idx] = (solution[i_idx] / toggle_mtx[i_idx, i]) % 2

        elif solution[i_idx] == 0:  # Solving 0x = 0 which has two solutions in Z2: 0 and 1.
            solution[i_idx] = 0

        else:
            raise ValueError("No solution exists.")

    # Format solution
    solution = np.array([solution[row_map[i]] for i in range(no_switches)]).reshape(dim)

    if transposed:
        solution = solution.T
        init_state = init_state.T

    return solution, init_state


# =================================================================== #
#                              Execution                              #
# =================================================================== #

if __name__ == "__main__":

    for n in range(1, 101):
        avg = 0

        for sims in range(5):
            start = time.time()
            puzzle = Puzzle(init_state=np.zeros((n, n)))
            puzzle.solve(final_state=1)
            delta = time.time() - start
            avg = (avg * sims + delta) / (sims + 1)

        print(f"{n}: {avg}")

        with open('runtimes.csv', 'a') as f:
            w = csv.writer(f)
            w.writerows([[n, avg]])
