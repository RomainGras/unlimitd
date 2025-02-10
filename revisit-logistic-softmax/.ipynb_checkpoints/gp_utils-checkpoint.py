import torch


# Training likelihood computations

def square_ntk(jac1, shrk_prms = None):
    """
    Computes the NTK used for training (always a squared matrix, one gradient computation needed)
    `jac1` (Tensor):jacobian of the network
    `shrk_prms` (Tensor) :Shrinking parameters
    """
    
    if shrk_prms is None:
        ntk_jac1 = [j.flatten(2) for j in jac1.values()]
    else:
        # Compute Sigma * J(x1)
        ntk_jac1 = [(shrk_prms[k]*j).flatten(2) for (k, j) in jac1.items()]   # Useful for the NTK computation
    
    # Compute J(x1) @ J(x2).T
    ntk = torch.stack([torch.einsum('Naf,Maf->aNM', j1, j2) for j1, j2 in zip(ntk_jac1, ntk_jac1)])
    ntk = ntk.sum(0)
    
    return ntk


def NMLL(phi, ntk, y_support, noise=.1):
    """
    Computes the negative marginal likelihood of one class's gaussian
    `phi` (Tensor) :output of the network at support inputsof class c
    `ntk` (Tensor) :ntk at the support inputs of class c
    `y_support` :(Tensor) :support labels
    `noise` :(float) :noise of the Gaussian process
    """
    X = y_support - phi
    
    # Perform Cholesky decomposition to get L
    # Noise
    n = ntk.size(0)
    device = f'cuda:{ntk.get_device()}'
    I = torch.eye(n, device=device)
    L = psd_safe_cholesky(ntk + noise*I)
    X = X.unsqueeze(1)
    
    # Solve L * Z = X for Z using forward substitution
    Z = torch.linalg.solve_triangular(L, X, upper=False)
    
    # Computes Z^T * Z = (Y-phi)^T ntk^-1 (Y-phi)
    sol1 = Z.T @ Z
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))
    
    return sol1 + logdet




# Meta-Testing predictive distribution computations

def support_query_ntk(jac1, jac2, shrk_prms = None):
    """
    Computes the NTK used for testing / Adaptation
    `jac1` (Tensor) :jacobian of the network with support inputs
    `jac2` (Tensor) :jacobian of the network with query inputs
    `shrk_prms` (Tensor) :Shrinking parameters
    """
    
    if shrk_prms is None:
        ntk_jac1 = [j.flatten(2) for j in jac1.values()]
        ntk_jac2 = [j.flatten(2) for j in jac2.values()]   # Useful for the NTK computation
    
    else:
        ntk_jac1 = [(shrk_prms[k]*j).flatten(2) for (k, j) in jac1.items()]   # Useful for the NTK computation
        ntk_jac2 = [(shrk_prms[k]*j).flatten(2) for (k, j) in jac2.items()]   # Useful for the NTK computation
    
    # Compute J(x1) @ J(x2).T
    ntk = torch.stack([torch.einsum('Naf,Maf->aNM', j1, j2) for j1, j2 in zip(ntk_jac1, ntk_jac2)])
    ntk = ntk.sum(0)
    
    return ntk

    
def out_distr(phi_support, phi_query, ntk_ss, ntk_sq, ntk_qq, y_support, var=False, noise=.1):
    """
    Computes the adapted mean and var (optional)
    `phi_support` (Tensor) :output of the network at support inputs of class c
    `phi_query` (Tensor) :output of the network at query inputs of class c
    `ntk_ss` (Tensor) :ntk at support - support inputs of class c
    `ntk_qs` (Tensor) :ntk at the query - support inputs of class c
    `ntk_qq` (Tensor) :ntk at the query - query inputs of class c
    `y_support` :(Tensor) :support labels
    `var` (bool) :Returns the variance of the distribution if True
    """
    X = y_support - phi_support
    
    # Perform Cholesky decomposition to get L
    n = ntk_ss.size(0)
    I = torch.eye(n, device='cuda')
    L = psd_safe_cholesky(ntk_ss + noise*I)
    X = X.unsqueeze(1)
    
    # Solve L * Z = X for Z using forward substitution
    Z1 = torch.linalg.solve_triangular(L, X, upper=False)
    Z2 = torch.linalg.solve_triangular(L, ntk_sq, upper=False)
    
    # Computes Z^T * Z = (Y-phi)^T ntk^-1 (Y-phi)
    mean = (Z2.T @ Z1).squeeze(1) + phi_query
    
    if var:
        var = ntk_qq - Z2.T @ Z2
        return (mean, var)
    
    else:
        return (mean,)
    
    
    
    
# To handle Cholesky decompositions

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        if A.dim() == 2:
            L = torch.linalg.cholesky(A, upper=upper, out=out)
            return L
        else:
            L_list = []
            for idx in range(A.shape[0]):
                L = torch.linalg.cholesky(A[idx], upper=upper, out=out)
                L_list.append(L)
            return torch.stack(L_list, dim=0)
    except:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(8):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                if Aprime.dim() == 2:
                    L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                    warnings.warn(
                        f"A not p.d., added jitter of {jitter_new} to the diagonal",
                        RuntimeWarning,
                    )
                    return L
                else:
                    L_list = []
                    for idx in range(Aprime.shape[0]):
                        L = torch.linalg.cholesky(Aprime[idx], upper=upper, out=out)
                        L_list.append(L)
                    warnings.warn(
                        f"A not p.d., added jitter of {jitter_new} to the diagonal",
                        RuntimeWarning,
                    )
                    return torch.stack(L_list, dim=0)
            except:
                continue

                

def safe_cholesky_solve(A, X):
    """
    Solve the inverse problem A^{-1} X, using psd_safe_cholesky to have good values for jitter
    """
    # Step 1: Perform Cholesky decomposition to get L
    L = psd_safe_cholesky(A)
    X = X.unsqueeze(1)
    
    # Step 2: Solve L * Z = X for Z using forward substitution
    Z = torch.linalg.solve_triangular(L, X, upper=False)  # 'upper=False' indicates L is lower triangular
    
    # Step 3: Solve L^T * Y = Z for Y using backward substitution
    Y = torch.linalg.solve_triangular(L.T, Z, upper=True)  # 'upper=True' indicates L^T is upper triangular

    return Y.squeeze()
