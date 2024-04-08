
from torch import *
import numpy as np

"""
Some basic functions for signal processing
Ref: https://github.com/sergiocallegari/PyDSM/blob/master/pydsm
"""
def acorr(x, N): 
    """
    Computes the raw autocorrelation of a vector up to lag N.

    Parameters
    ----------
    x : array_like
        1-D sequence to compute the auto-correlation upon
    N : int
        the maximum (positive) lag of the raw auto-correlation to return.

    Returns
    -------
    q : ndarray
        the raw (unnormalized) autocorrelation vector.
        Assuming that m is the length of x
        q(k) = sum_{n=k}^{m-1} x(n) x(n-k) for k = 0 ... N

    Notes
    -----
    The routine does not make any check on the length of x and N. It
    is responsibility of the user to assure that len(x)>=N. In some cases
    (but only in some cases), zero padding is practiced.
    """
    m = len(x)
    if not is_tensor(x):
        x = tensor(x)
    
    q = as_tensor([torch.dot(x[k:m], x[0:m-k]) for k in range(N+1)])
    return q


def xcorr(x, y, N):
    """
    Computes the raw crosscorrelation between two vectors up to lag N.

    Parameters
    ----------
    x : array_like
        first 1-D vector
    y : array_like
        second 1-D vector
    N : int
        the maximum (positive) lag of the raw cross-correlation to return.

    Returns
    -------
    q : ndarray
        the raw (unnormalized) crosscorrelation vector.
        Assuming that mx and my are the lengths of x and y
        q(k) = sum_{n=k}^{min(mx-1,my+k-1)} x(n) y(n-k) for k = 0 ... N

    Notes
    -----
    the routine does not make any check on the lengths of x, y and N. It
    is responsibility of the user to assure that N<=len(y). In some cases
    (but only in some cases), zero padding is assumed.
    """
    mx = len(x)
    my = len(y)
    q = as_tensor([dot(y[k:min(my, mx+k)],
                           x[0:min(my-k, mx)]) for k in range(N+1)])
    return q


def shiftdim(x, n=None, nargout=2):
    """
    Shift dimensions a la Matlab

    When n is provided, shiftdim shifts the axes of x by n.
    If n is positive, it shifts the axes to the left, wrapping the
    leading axes with non unitary length to the end.
    When n is negative, it shifts the axes to the right, inserting n leading
    axes with unitary length.
    When n is not provided or None, it shifts the axes to the left, reducing
    the number of dimensions and removing all the leading axes with unitary
    length.

    Parameters
    ----------
    x : array like
        multi-dimensional array to operate upon
    n : int or None, optional
        amount to shift. Defaults to None, which means automatic computation
    nargout : int
        number of output values

    Returns
    -------
    y : ndarray
        the result of the axes shift operation
    n : int
        the actual shift

    Examples
    --------
    >>> from numpy.random import rand
    >>> a = rand(1, 1, 3, 1, 2)
    >>> b, n = shiftdim(a)
    >>> tensor(b.size())
    tensor([3, 1, 2])
    >>> n
    2
    >>> c = shiftdim(b, -n, nargout=1)
    >>> torch.all(c == a)
    True
    >>> d = shiftdim(a, 3, nargout=1)
    >>> tensor(d.size())
    tensor([1, 2, 1, 1, 3])

    >>> b, n = shiftdim([[[1]]])
    >>> b, n
    (tensor([[[1]]]), 0)
    """
    outsel = slice(nargout) if nargout > 1 else 0
    x = torch.as_tensor(x)
    s = tuple(x.size())
    m = next((i for i, v in enumerate(s) if v > 1), 0)
    if n is None:
        n = m
    if n > 0:
        n = n % x.dim()
    if n > 0:
        if n <= m:
            x = x.reshape(s[n:])
        else:
            x = x.transpose(roll(range(x.dim()), -n))
    elif n < 0:
            x = x.reshape((1)*(-n) + s)
    return (x, n)[outsel]


def apply_along_axis(function, x, axis=0, *args, **kwargs):
    """
    apply a function along a given axis
    reimplemented from numpy.apply_along_axis
    
     Execute `func1d(a, *args, **kwargs)` where `func1d` operates on 1-D arrays
    and `a` is a 1-D slice of `arr` along `axis`.

    Parameters
    ----------
    func1d : function (M,) -> (Nj...)
        This function should accept 1-D arrays. It is applied to 1-D
        slices of `arr` along the specified axis.
    axis : integer
        Axis along which `arr` is sliced.
    arr : ndarray (Ni..., M, Nk...)
        Input array.
    args : any
        Additional arguments to `func1d`.
    kwargs : any
        Additional named arguments to `func1d`.
        

    Returns
    -------
    out : ndarray  (Ni..., Nj..., Nk...)
        The output array. The shape of `out` is identical to the shape of
        `arr`, except along the `axis` dimension. This axis is removed, and
        replaced with new dimensions equal to the shape of the return value
        of `func1d`. So if `func1d` returns a scalar `out` will have one
        fewer dimensions than `arr`.
    """
    # handle negative axis
    return stack([
        function(x_i, *args, **kwargs) for x_i in torch.unbind(x, dim=axis)
    ], dim=axis)
    

            
def cplxpair(x, tol=None, dim=None):
    """
    Sorts values into complex pairs a la Matlab.

    The function takes a vector or multidimensional array of of complex
    conjugate pairs or real numbers and rearranges it so that the complex
    numbers are collected into matched pairs of complex conjugates. The pairs
    are ordered by increasing real part, with purely real elements placed
    after all the complex pairs.

    In the search for complex conjugate pairs a relative tolerance equal to
    ``tol`` is used for comparison purposes. The default tolerance is
    100 times the system floating point accuracy.

    If the input vector is a multidimensional array, the rearrangement is done
    working along the axis specifid by the parameter ``dim`` or along the
    first axis with non-unitary length if ``dim`` is not provided.

    Parameters
    ----------
    x : array_like of complex
        x is an array of complex values, with the assumption that it contains
        either real values or complex values in conjugate pairs.
    tol: real, optional
        relative tolerance for the recognition of pairs.
        Defaults to 100 times the system floating point accuracy for the
        specific number type.
    dim: integer, optional
        The axis to operate upon.

    Returns
    -------
    y : ndarray
        y is an array of complex values, with the same values in x, yet now
        sorted as complex pairs by increasing real part. Real elements in x
        are place after the complex pairs, sorted in increasing order.

    Raises
    ------
    ValueError
        'Complex numbers cannot be paired' if there are unpaired complex
        entries in x.

    Examples
    --------
    >>> a = np.exp(2j*np.pi*np.arange(0, 5)/5)
    >>> b1 = cplxpair(a)
    >>> b2 = torch.as_tensor([-0.80901699-0.58778525j, -0.80901699+0.58778525j,
    ...                   0.30901699-0.95105652j,  0.30901699+0.95105652j,
    ...                   1.00000000+0.j])
    >>> torch.allclose(b1, b2)
    True

    >>> cplxpair(1)
    tensor([1])

    >>> cplxpair([[5, 6, 4], [3, 2, 1]])
    tensor([[3, 2, 1],
           [5, 6, 4]])

    >>> cplxpair([[5, 6, 4], [3, 2, 1]], dim=1)
    tensor([[4, 5, 6],
           [1, 2, 3]])

    """
    def cplxpair_vec(x, tol):
        real_mask = np.abs(x.imag) <= tol*np.abs(x)
        x_real = np.sort(np.real(x[real_mask]))
        x_cplx = np.sort(x[np.logical_not(real_mask)])
        if x_cplx.size == 0:
            return x_real
        if (x_cplx.size % 2) != 0:
            raise ValueError('Complex numbers cannot be paired')
        if np.any(np.real(x_cplx[1::2])-np.real(x_cplx[0::2]) >
                  tol*np.abs(x_cplx[0::2])):
            raise ValueError('Complex numbers cannot be paired')
        start = 0
        while start < x_cplx.size:
            sim_len = next((i for i, v in enumerate(x_cplx[start+1:]) if
                           (np.abs(np.real(v)-np.real(x_cplx[start])) >
                            tol*np.abs(v))), x_cplx.size-start-1)+1
            if (sim_len % 2) != 0:
                sim_len -= 1
            # At this point, sim_len elements with identical real part
            # have been identified.
            sub_x = x_cplx[start:start+sim_len]
            srt = np.argsort(np.imag(sub_x))
            sub_x = sub_x[srt]
            if np.any(np.abs(np.imag(sub_x)+np.imag(sub_x[::-1])) >
                      tol*np.abs(sub_x)):
                raise ValueError('Complex numbers cannot be paired')
            # Output should contain "perfect" pairs. Hence, keep entries
            # with positive imaginary parts amd use conjugate for pair
            x_cplx[start:start+sim_len] = np.concatenate(
                (np.conj(sub_x[:sim_len//2-1:-1]),
                 sub_x[:sim_len//2-1:-1]))
            start += sim_len
        return np.concatenate((x_cplx, x_real))

    x = np.atleast_1d(x)
    if x.size == 0:
        return x
    if dim is None:
        dim = next((i for i, v in enumerate(x.shape) if v > 1), 0)
    if tol is None:
        try:
            tol = 100*np.finfo(x.dtype).eps
        except:
            tol = 100*np.finfo(x.dtype).eps
    output = np.apply_along_axis(cplxpair_vec, dim, x, tol)
    return as_tensor(output)