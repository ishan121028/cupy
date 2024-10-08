import unittest
import cupy
import numpy as np
from cupyx.scipy import sparse


# Helper function to create sparse matrix
def _make_sparse_matrix(dtype):
    # Simple sparse matrix 3x3 with a few non-zero entries
    data = np.array([1, 2, 3], dtype=dtype)
    row = np.array([0, 1, 2])
    col = np.array([0, 1, 2])
    return sparse.csr_matrix((data, (row, col)), shape=(3, 3))


class TestMatrixIO(unittest.TestCase):

    def setUp(self):
        # Prepare a sparse matrix for testing
        self.dtype = np.float32
        self.sparse_matrix = _make_sparse_matrix(self.dtype)

    def test_load_npz(self):
        """Test loading a sparse matrix from an npz file."""
        # Save the sparse matrix to a temporary npz file
        sparse.save_npz("test_matrix.npz", self.sparse_matrix)

        # Load the matrix from the npz file
        loaded_matrix = sparse.load_npz("test_matrix.npz")

        # Verify that the loaded matrix matches the original
        cupy.testing.assert_array_equal(
            loaded_matrix.toarray(), self.sparse_matrix.toarray()
        )


if __name__ == "__main__":
    unittest.main()
