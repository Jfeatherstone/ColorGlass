import numpy as np

import numba

from itertools import product

"""
The expm calculations has to be performed fast; scipy does not provide required speed,
so we instead implement a version using the numba library.

Modified version of:
https://github.com/michael-hartmann/expm  
"""


# Relevant values from Table 10.2 in Functions of Matrices, Higham (2008)
# Defines the values used to determine the optimal degree of the
# calculation
theta3  = 1.5e-2
theta5  = 2.5e-1
theta7  = 9.5e-1
theta9  = 2.1e0
theta13 = 5.4e0

# And the coefficients for equation 10.33 corresponding to the degree
# values in the above dictionary. The number of coefficients is the degree+1 (eg. b3 has 3+1 elements)
b3  = [120,60,12,1]
b5  = [30240,15120,3360,420,30,1]
b7  = [17297280, 8648640,1995840, 277200, 25200,1512, 56,1]
b9  = [17643225600,8821612800,2075673600,302702400,30270240, 2162160,110880,3960,90,1]
b13 = [64764752532480000,32382376266240000,7771770303897600,1187353796428800,129060195264000,10559470521600,670442572800,33522128640,1323241920,40840800,960960,16380,182,1]


@numba.jit(nopython=True) 
def _expm_pade(A,M):
    """
    Compute the exponential of a square matrix using the Pade approximation, as described in
    _Functions of Matrices: Theory and Computation_i [1], algorithm 10.3, lines 1-6.

    Parameters
    ----------

    A : numpy.mat
        A square matrix to the compute the exponential of

    M : positive integer
        The degree of the matrix calculation (see [1])
    """
    dtype = A.dtype
    dim, dim = A.shape
   
    # The coefficients for solving equation 10.33 from [1] based on the degree
    # Unfortunately, numba doesn't play well with dictionaries, so we have to do
    # this the ugly way
    if M == 3:
        b = [120,60,12,1]
    elif M == 5:
        b = [30240,15120,3360,420,30,1]
    elif M == 7:
        b = [17297280, 8648640,1995840, 277200, 25200,1512, 56,1]
    elif M == 9:
        b = [17643225600,8821612800,2075673600,302702400,30270240, 2162160,110880,3960,90,1]
    elif M == 13:
        b = [64764752532480000,32382376266240000,7771770303897600,1187353796428800,129060195264000,10559470521600,670442572800,33522128640,1323241920,40840800,960960,16380,182,1]

    # The list of supported numpy functions can be found here:
    # https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
    # It lists np.identity as being supported, but throws an error whenever
    # you actually try to use it, so eye is entirely equivalent and _does_ work
    U = b[1]*np.eye(dim, dtype=dtype)
    V = b[0]*np.eye(dim, dtype=dtype)

    A2 = np.dot(A,A)
    A2n = np.eye(dim, dtype=dtype)

    # evaluate (10.33)
    for i in range(1,M//2+1):
        A2n = np.dot(A2n,A2)
        U += b[2*i+1]*A2n
        V += b[2*i]  *A2n

    #del A2,A2n
    U = np.dot(A,U)

    return np.linalg.solve(V-U, V+U)


@numba.jit(nopython=True) 
def _expm_ss(A,norm):
    """
    Compute the exponential of a square matrix using the scaling and squaring method, as described in
    _Functions of Matrices: Theory and Computation_i [1], algorithm 10.3, lines 7-13.

    Parameters
    ----------

    A : numpy.mat
        A square matrix to the compute the exponential of

    norm : double
        The norm of the matrix A
        
    """

    dim, dim = A.shape

    # The coefficients for m=13 used to solve equation 10.33 from [1]
    b = [64764752532480000,32382376266240000,7771770303897600,1187353796428800,129060195264000,10559470521600,670442572800,33522128640,1323241920,40840800,960960,16380,182,1]

    # Ensure that A/2^s <= theta13
    s = max(0, int(np.ceil(np.log(norm/theta13)/np.log(2))))
    if s > 0:
        A /= 2**s

    Id = np.eye(dim)
    A2 = np.dot(A,A)
    A4 = np.dot(A2,A2)
    A6 = np.dot(A2,A4)

    U = np.dot(A, np.dot(A6, b[13]*A6+b[11]*A4+b[9]*A2)+b[7]*A6+b[5]*A4+b[3]*A2+b[1]*Id)
    V = np.dot(A6, b[12]*A6+b[10]*A4+b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*Id

    # We'll get a warning from numba that dot is slow on non-contiguous arrays here if
    # we don't convert first. Not sure how much it actually speeds anything up, but
    # it gets rid of the warning. (NOTE: without conversion, array is of type 'F')
    r13 = np.ascontiguousarray(np.linalg.solve(V-U, V+U))

    return np.linalg.matrix_power(r13, 2**s)

@numba.jit(nopython=True) 
def expm(A):
    r"""
    Calculate the matrix exponential of a square matrix A

    This module implements algorithm 10.20 from _Functions of Matrices: Theory and Computation_ [1].
    The matrix exponential is calculated using either the scaling and squaring method or the Pade approximation,
    depending on which will be more accurate.

    Per [1], the error on the calculation is of order \(10^{-16}\).

    References
    ----------

    [1] Higham, N. J. (2008). _Functions of matrices: Theory and computation_. Society for Industrial and Applied Mathematics.

    """
    # Calculate the norm of A
    norm = np.linalg.norm(A, ord=1)

    # Decide which numerical method to use given the norm of the matrix
    if   norm < theta3:
        return _expm_pade(A,3)
    elif norm < theta5:
        return _expm_pade(A,5)
    elif norm < theta7:
        return _expm_pade(A,7)
    elif norm < theta9:
        return _expm_pade(A,9)
    else:
        return _expm_ss(A,norm)


"""
The following two methods are used to generate a set of
basis matrices used in calculating the Wilson Line

Modified version of:
https://github.com/CQuIC/pysme/blob/master/src/pysme/gellmann.py
originally made by Jonathan Gross
"""

def gen_gellmann(j, k, d):
    r"""
    Returns a generalized Gell-Mann matrix of dimension d according to Bertlmann & Krammer (2008).

    Adapted from Jonathan Gross's [pysme library](https://github.com/CQuIC/pysme/blob/master/src/pysme/gellmann.py).

    For generation of all generalized Gell-Mann matrices for a given dimension, see `cgc.LinAlg.get_basis`.


    Parameters
    ----------
    j : positive integer
        Index for generalized Gell-Mann matrix
    k : positive integer
        Index for generalized Gell-Mann matrix
    d : positive integer
        Dimension of the generalized Gell-Mann matrix

    Returns
    -------
    numpy.array
        A genereralized Gell-Mann matrix

    References
    ----------

    Bertlmann, R. A., & Krammer, P. (2008). Bloch vectors for qudits. Journal of Physics A: Mathematical and Theoretical, 41(23), 235303. [10.1088/1751-8113/41/23/235303](https://doi.org/10.1088/1751-8113/41/23/235303)
    """

    if j > k:
        gjkd = np.zeros((d, d), dtype='complex')
        gjkd[j - 1][k - 1] = 1
        gjkd[k - 1][j - 1] = 1
    elif k > j:
        gjkd = np.zeros((d, d), dtype='complex')
        gjkd[j - 1][k - 1] = -1.j
        gjkd[k - 1][j - 1] = 1.j
    elif j == k and j < d:
        gjkd = np.sqrt(2/(j*(j + 1)))*np.diag([1 + 0.j if n <= j
                                               else (-j + 0.j if n == (j + 1)
                                                     else 0 + 0.j)
                                               for n in range(1, d + 1)])
    else:
        gjkd = np.diag([1 + 0.j for n in range(1, d + 1)])*np.sqrt(2/d)

    return gjkd


def get_basis(d):
    r"""Return a Hermitian and traceless set of basis matrices for \(SU(d)\), as well
    as the identity. The former matrices satisfy:

    The basis is made up of \(d^2 - 1\) generalized Gell-Mann matrices, and then the identity
    as the last matrix.

    $$ tr( t^a t^b) = \frac{1}{2} \delta_{ab} $$

    For individual generation information, see `cgc.LinAlg.gen_gellmann`

    Parameters
    ----------

    d : positive integer
        The dimension of the Hilbert space

    Returns
    -------

    list of numpy.ndarray
        The basis matrices

    """

    return np.array([gen_gellmann(j, k, d)/2 for j, k in product(range(1, d + 1), repeat=2)])
