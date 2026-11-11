import torch

from src.kernels import rbf_kernel, affine_kernel, constant_kernel, safe_cholesky


def test_rbf_kernel_diagonal_is_one():
    x = torch.rand(5, 2)
    ell = torch.ones(2)
    K = rbf_kernel(x, x, ell)
    assert torch.allclose(torch.diagonal(K), torch.ones(5), atol=1e-6)


def test_rbf_kernel_symmetric_and_psd():
    x = torch.rand(10, 3)
    ell = torch.rand(3) + 0.1
    K = rbf_kernel(x, x, ell)
    assert torch.allclose(K, K.T, atol=1e-6)
    eigvals = torch.linalg.eigvalsh(K + 1e-6 * torch.eye(10))
    assert (eigvals > -1e-5).all()


def test_rbf_kernel_decreases_with_distance():
    ell = torch.ones(1)
    x0 = torch.zeros(1, 1)
    x_near = torch.full((1, 1), 0.1)
    x_far = torch.full((1, 1), 2.0)
    k_near = rbf_kernel(x0, x_near, ell).item()
    k_far = rbf_kernel(x0, x_far, ell).item()
    assert k_near > k_far


def test_affine_kernel_matches_formula():
    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 4.0]])
    K = affine_kernel(x1, x2)
    assert torch.allclose(K, torch.tensor([[1.0 + 1 * 3 + 2 * 4]]))


def test_constant_kernel_all_ones():
    K = constant_kernel(3, 4)
    assert torch.allclose(K, torch.ones(3, 4))


def test_safe_cholesky_succeeds_on_near_singular_matrix():
    n = 6
    rank1 = torch.ones(n, n)  # singular (rank 1)
    chol, jitter = safe_cholesky(rank1, 1e-7, 10.0, 10)
    assert jitter >= 1e-7
    reconstructed = chol @ chol.T
    assert torch.allclose(reconstructed, rank1, atol=1e-2)
