import torch
import time
from torch.func import functional_call, vmap, vjp, jvp, jacrev



def create_random_projection_matrix(n, subspace_dimension):
    """
    Create a projection matrix from R^n to a subspace of dimension `subspace_dimension`.
    
    Args:
    n (int): Dimension of the original space.
    subspace_dimension (int): Dimension of the target subspace.

    Returns:
    torch.Tensor: A (n x subspace_dimension) projection matrix.
    """
    # Check if subspace_dimension is not greater than n
    if subspace_dimension > n:
        raise ValueError("subspace_dimension must be less than or equal to n")

    # Generate a random n x subspace_dimension matrix
    random_matrix = torch.randn(n, subspace_dimension)

    # Perform QR decomposition to orthonormalize the columns
    q, _ = torch.linalg.qr(random_matrix)

    # Return the first 'subspace_dimension' columns of Q, which form an orthonormal basis
    return q[:, :subspace_dimension].T




def low_rank_approx(Y, W, psi):
    """
    given Y = A @ Om, (N, k)
    and W = Psi @ A, (l, M)
    and Psi(X) = Psi @ X, (N,...) -> (l,...)
    where Om and Psi and random sketching operators
    returns Q (N x k), X (k x M) such that A ~= QX
    """
    # Perform QR decomposition on Y to get orthonormal basis Q
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    
    # Apply Psi to Q and then perform QR decomposition
    U, T = torch.linalg.qr(psi@Q, mode='reduced')
    
    # Solve the triangular system T @ X = U^T @ W for X
    # PyTorch does not have a direct equivalent to scipy.linalg.solve_triangular,
    # so we use torch.linalg.solve which can handle triangular matrices if specified.
    X = torch.linalg.solve_triangular(T, U.T@ W, upper=False)
    
    return Q, X


def sym_low_rank_approx(Y, W, psi):
    """
    Perform a symmetric low-rank approximation of the matrix A.
    """
    Q, X = low_rank_approx(Y, W, psi)  # Assuming Psi is now correctly handled
    k = Q.shape[-1]  # Dimension of the sketches
    
    # Concatenate Q and X.T along columns to form a larger matrix
    tmp = torch.cat((Q, X.T), dim=1)  # Correctly access the transpose
    
    # Perform QR decomposition on the concatenated matrix
    U, T = torch.linalg.qr(tmp, mode='reduced')
    
    # Extract T1 and T2 from T
    T1 = T[:, :k]
    T2 = T[:, k:2*k]
    
    # Compute symmetric matrix S
    S = (T1 @ T2.T + T2 @ T1.T) / 2
    
    return U, S


def fixed_rank_eig_approx(Y, W, psi, r):
    """
    Returns U (N x r), D (r) such that A ~= U diag(D) U^T using PyTorch.
    """
    # Obtain symmetric low-rank approximation
    U, S = sym_low_rank_approx(Y, W, psi)
    
    # Compute eigenvalues and eigenvectors
    D, V = torch.linalg.eigh(S)
    
    # Truncate to keep the top-r eigenvalues and corresponding eigenvectors
    D = D[-r:]  # Top r eigenvalues
    V = V[:, -r:]  # Corresponding eigenvectors
    
    # Update U to be U @ V
    U = U @ V
    
    return U, D


def sketch(net, batches, k, l, c):
    """
    Returns a good rank 2k approximation of the FIM using PyTorch.
    """
    M = batches.size(0)
    N_params = sum(p.numel() for p in net.parameters())
    print(N_params)

    om = torch.randn(k, N_params).cuda()
    psi = torch.randn(l, N_params).cuda()

    Y = torch.zeros(N_params, k).cuda()
    W = torch.zeros(l, N_params).cuda()

    for batch in batches:
        J = compute_jacobian_vmap_autodiff(net, batch, c)
        Y += (om @ J.T @ J).T / M
        W += (psi @ J.T @ J) / M

    # Compute the rank-2k approximation
    U, D = fixed_rank_eig_approx(Y, W, psi, 2 * k)

    return U, D

def jacobian_autodiff(net, batch, c):
        """
        Return the jacobian of a batch of inputs, using autodifferentiation
        Useful for when dealing with models using batch normalization or other kind of running statistics
        """
        batch.requires_grad_(True)
        net.parameters()
        outputs = net(batch).requires_grad_(True)
        N = sum(p.numel() for p in net.parameters())
        jac = torch.empty(outputs.size(0), N).to("cuda:0")
        for j in range(outputs.size(0)):
            # print(j)
            grad_y1 = torch.autograd.grad(outputs[j, c], net.parameters()) # We need to create and retain every single graph for the gradient to be able to run through during backprop
            # print_memory_usage()
            flattened_tensors = [t.flatten() for t in grad_y1]
            jac[j] = torch.cat(flattened_tensors)
            # print_memory_usage()
            # if device == "cuda":
            #     torch.cuda.empty_cache()
            #     print_memory_usage()
        return jac

def compute_jacobian_vmap_autodiff(net, batch, c):
    """
    Return the jacobian of a batch of inputs, thanks to the vmap functionality
    """
    device = "cuda"
    if device=='cuda':
        torch.cuda.empty_cache()
    params_that_need_grad = []
    for param in net.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)

    batch = batch.to(device, non_blocking=True)
    batch.requires_grad_(True)
    outputs = net(batch)
    basis_vectors = torch.eye(len(batch),device=device,dtype=torch.bool)
    J_layer = []
    for i,z in enumerate(net.named_parameters()):
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2} MB")
        if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
            continue
        name, param = z
        outputsc = outputs[:, c]   
        #Seems like for retain_graph=False, you might need to do multiple forward passes.

        def torch_row_Jacobian(v): #y would have to be a single piece of the batch
            return torch.autograd.grad(outputsc,param,v, retain_graph=False)[0].reshape(-1)
        J_layer.append(vmap(torch_row_Jacobian)(basis_vectors).detach())

        del outputsc
        if device=='cuda':
            torch.cuda.empty_cache()
        #print(name)
    #for layer in J_layer:
    #    print(layer.shape)
    del params_that_need_grad
    del outputs
    del basis_vectors
    if device=='cuda':
        torch.cuda.empty_cache()
    J_layer = torch.cat(J_layer, axis=1)
    return J_layer


def proj_sketch(net, batches, subspace_dimension, c):
    t = time.time_ns()
    
    T = 6 * subspace_dimension + 4 
    k = (T - 1) // 3                    # k = 2 * subspace_dimension + 1
    l = T - k                           # l = 4 * subspace_dimension + 3

    U, D = sketch(net, batches, k, l, c)
    idx = D.argsort(descending=True)
    print("U shape:", U.shape)
    print("Index tensor:", idx)
    print("Requested subspace dimension:", subspace_dimension)
    
    # Ensure idx is of type long for indexing
    # idx = idx.long()

    P1 = U[:, idx[:subspace_dimension]].T

    print(f"Done sketching in {(time.time_ns() - t) / 1e9:.4f} s")

    return P1
