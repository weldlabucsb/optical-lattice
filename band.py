##Code has for objective the calculation of the energy bands of a periodic potential V. 
#Author: Enrique Morell

import numpy as np
import scipy.constants as ct
import matplotlib.pyplot as plt
k = 1
m = 1
ER = ct.hbar**2*k**2/(2*m)
#jmax:
jmax = 1
def sin_potential(x, V0 = 100, k = 1):
    return V0*np.power(np.sin(k*x),2)

def fourier_series_coeff_numpy(f, T, N, return_complex=True):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag

Vj = fourier_series_coeff_numpy(sin_potential, np.pi/k, jmax)
print(f"V0 = {Vj[0]}, V1 = {Vj[1]}") # , V2 = {Vj[2]}, V3 = {Vj[3]}")
print(f"V0 = {Vj[0]}, V-1 = {np.conj(Vj[1])}")
Vj = np.pad(Vj, (0,jmax), 'constant', constant_values = (0,))
print(Vj.shape)


def central_eq_solv(q, jmax = 20):
    C = np.zeros((2*jmax+1, 2*jmax+1), dtype=complex)
    it = np.nditer(C, flags=['multi_index'], op_flags=['writeonly'])
    with it:
        while not it.finished:
            i, j = it.multi_index
            #if in the diagonal:
            if i == j:
                it[0] = (2*(j-jmax)+q/k)**2+Vj[0]
                #print("%d <%s>" % (2*(j-jmax)+q/k, it.multi_index), end=' ')
            else:
                V0 = Vj[i]
                if i<j: #we are above the diagonal
                    V0 = np.conj(Vj[i])
                it[0] = V0
                #print("%d <%s>" % (V0, it.multi_index), end=' ')
            it.iternext()
    print(C)
    return np.linalg.eigh(C)


w, v = central_eq_solv(0.5, jmax=jmax)    
print(w)

Q = np.linspace(-1*k, 1*k, 10)
E = []
for q in Q:
    w, v = central_eq_solv(q, jmax=jmax)
    E.append(np.sort(w))
E = np.array(E).T
print(E)
print(E.shape)
for band in E:
    plt.plot(Q, band)
plt.show()
