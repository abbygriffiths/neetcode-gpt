import numpy as np
from numpy.typing import NDArray


class Solution:
    # O(n) time | O(n) space
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        adjusted_z = z - np.max(z)
        exps = np.exp(adjusted_z)
        return np.round(exps / np.sum(exps), 4)
