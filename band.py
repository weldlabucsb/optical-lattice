##Code has for objective the calculation of the energy bands of a periodic potential V. 
#Author: Enrique Morell

import numpy as np
import scipy.constants as ct
import matplotlib.pyplot as plt

class BandSolver:
    ###Class intended to contain all the tools needed for solving band problems.
    def fourier_series_coeff_1D(self, f, T, N, return_complex=True):
        """
        Credit for this method to gg349 from Stack Overflow (https://stackoverflow.com/a/27720302)
        Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

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
    def fourier_series_coeff_2D(self, f, Tx, Ty, N, return_complex=True):
        """
        Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

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
        f : the periodic function, a callable like f(t), needs to be compatible with numpy.meshgrid, i.e. f(np.array) needs to work correctly
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
        f_sample = 2 * N
        t1, dt = np.linspace(0, Tx, f_sample+2, endpoint=False, reststep=True)
        t2, dt = np.linspace(0, Ty, f_sample+2, endpoint=False, retstep=True)

        T = np.meshgrid(t1, t2)
        y = np.fft.rfft2(f(T), norm="ortho")
        if return_complex:
            return y #in the algorithm for 2D we use all with negative indices so we need the conjugate
        else:
            return "not implemented"





    def __init__(self, potential, jmax = 20, kx = 1, ky = 1, kz = 1, dim = 1, debug = False):
        self.dim = dim
        self.potential = potential #potential should be a function of dimension given
        self.jmax = jmax
        if dim ==1:
            self.k = kx
        if dim >= 2:
            self.kx = kx
            self.ky = ky
            self.k =  np.sqrt(kx**2+ky**2)

            self.kvec = np.array([self.kx, self.ky])
            self.kmag = np.linalg.norm(self.kvec)
            self.kmag_sq = self.kmag**2
        if dim == 3:
            self.kz = kz

            self.kvec = np.array([self.kx, self.ky, self.kz]) #avoids recalculation for big matrices.
            self.kmag = np.linalg.norm(self.kvec)
            self.kmag_sq = self.kmag**2
            
        
        self.debug = debug

        self.error = False
        self.errormessage = ""

        #calculates fourier coefficients used later.
        self.fourier()
    def fourier(self):
        if self.dim == 1:
            self.Vj = self.fourier_series_coeff_1D(self.potential, np.pi/self.k, self.jmax)
            self.Vj = np.pad(self.Vj, (0,self.jmax), 'constant', constant_values = (0,))
        if self.dim == 2:
            self.Vj = self.fourier_series_coeff_2D(self.potential, np.pi/self.kx, np.pi/self.ky, self.jmax)
            self.Vj = np.pad(self.Vj, (0, self.jmax), 'constant', constant_values = (0,))
            
        if self.dim > 2:
            print("Higher dimension than 1 is not yet implemented. Please try again another time.")
            #TODO: replace this with a proper error message native to Python.
            self.error = True
            self.errormessage = "Higher dimension than 1 is not yet implemented. Please try again another time."

        if self.debug:
            if not self.error:
                print(f"V0 = {self.Vj[0]}, V1 = {self.Vj[1]}") # , V2 = {Vj[2]}, V3 = {Vj[3]}")
                print(f"V0 = {self.Vj[0]}, V-1 = {np.conj(self.Vj[1])}")
                print(f"Shape of Vj vector after padding: {self.Vj.shape}")
            else:
                print(self.errormessage)
        
    def _solve1D(self, q):
        C = np.zeros((2*self.jmax+1, 2*self.jmax+1), dtype=complex) #empty matrix which will represent the bloch's equation.
        it = np.nditer(C, flags=['multi_index'], op_flags=['writeonly']) #allows for a more efficient iteration through the matrix.
        with it:
            while not it.finished:
                i, j = it.multi_index #{i,j} represents the row and column currently being read and writen.
                #if in the diagonal:
                if i == j:
                    it[0] = (2*(j-self.jmax)+q/self.k)**2+self.Vj[0] #comes from the 1D periodic bloch equation check Lab Book #1 p. 108 by Enrique Morell
                else:
                    V0 = self.Vj[np.abs(j-i)] #the vertical distance to your diagonal gives you the index of the fourier coefficient to be put in there.
                    if i<j: #we are above the diagonal
                        V0 = np.conj(self.Vj[j-i])
                    it[0] = V0
                it.iternext() #next cell
        #print(C)
        return np.linalg.eigh(C) #returns the eigen values of the symmetric matrix.
    def _solve2D(self, q : np.ndarray):
        """
        q is a numpy array with qx and qy in each coordinate
        """
        C = np.zeros(((2*self.jmax+1)**2, (2*self.jmax+1)**2), dtype=complex)
        n = 2*self.jmax+1
        it = np.nditer(C, flags=['multi_index'], op_flags=['writeonly']) #allows for a more efficient iteration through the matrix.
        with it:
            while not it.finished:
                i, j = it.multi_index #{i, j} represents the coefficients of row, column
                if i == j:
                    #on the diagonal:
                    it[0] = self._a(i//n,(j-n)%(n), q)
                else:

                    V0 = self.Vj[i//n, (np.abs(j-i)-n)%n]
                    if i < j: #we are above the diagonal
                        V0 = np.conj(V0) #self.Vj[i//n, (j-i-n)%n]
                    it[0] = V0
                it.iternext() #next cell
        return np.linalg.eigh(C) #returns the eigen values of the symmetric matrix.
    def _a(j1,j2, q : np.array):
        qx, qy = q[0], q[1]
        # j1 -= self.jmax
        # j2 -= self.jmax
        return (qx/self.k+2*j1*self.kx/self.k)**2+(qy/self.k+2*j2*self.ky/self.k)**2+self.Vj[0,0] #in Er scale
    # def filler(self):
    #     C = np.zeros((2*self.jmax+1, 2*self.jmax+1, 2*self.jmax+1), dtype=complex)
    #     # C = np.zeros((2*self.jmax+1, 2*self.jmax+1), dtype=complex) #empty matrix which will represent the bloch's equation.
    #     # it = np.nditer(C, flags=['multi_index'], op_flags=['writeonly']) #allows for a more efficient iteration through the matrix.
    #     # with it:
    #     #     while not it.finished:
    #     #         i, j = it.multi_index #{i,j} represents the row and column currently being read and writen.
    #     #         #if in the diagonal:
    #     #         if i == j:
    #     #             it[0] = (2*(j-self.jmax)+q/self.k)**2+self.Vj[0] #comes from the 1D periodic bloch equation check Lab Book #1 p. 108 by Enrique Morell
    #     #         else:
    #     #             V0 = self.Vj[np.abs(j-i)] #the vertical distance to your diagonal gives you the index of the fourier coefficient to be put in there.
    #     #             if i<j: #we are above the diagonal
    #     #                 V0 = np.conj(self.Vj[j-i])
    #     #             it[0] = V0
    #     #         it.iternext()

    def solve(self, qmin, qmax, N = 100):
        """
        Calculates the bands of the given potential.
        Parameters
        ----------
        qmin: lower bound of q between which the band structure will be calculated. (remember q/k = -1, +1 is the FBZ!)
        qmax: upper bound of the band calculation.
        N: points in the calculation.
        Returns
        ----------
        Q: Numpy array containing the linearly spaced quasimomentum for which the eigen values where calculated.
        E: Numpy array containing the bands. E[0] is the ground band, E[1] the first excited band and so on.
        V: Numpy array containing the eigen vectors (Cj coefficients) of the wave function.
        """
        Q = np.linspace(qmin, qmax, N)
        E = []
        V = []
        for q in Q:
            if self.dim == 1:
                w, v = self._solve1D(q)
            if sel.dim == 2:
                w, v = self._solve2D(q)
            perm = np.argsort(w) #finds the permutation to sort w from smallest energy to highest
            E.append(w[perm]) #applies the permutation to both the eigen values and the eigen vectors.
            V.append(v[perm])
        return (Q, np.array(E).T, np.array(V).T) #transposes the arrays for easy plotting.

###Example of use:
Potential to be used.
def sin_potential(x, V0 = 12, k = 1):
    return V0*np.power(np.sin(k*x),2)
def sin_potential_2d(x, y, V0 =12, k=1):
    return V0*np.power(np.sin(k*x)*np.sin(k*y),2)
k = 1
m = 1
ER = ct.hbar**2*k**2/(2*m) #energy scale. It is implied in the class. All energies are in Er.
#jmax:
jmax = 10

band = BandSolver(sin_potential_2d, jmax=jmax, k=k, dim=2)
Q, E, V = band.solve(-1*k, 1*k)

#plotting the first five bands.
counter = 0
n=5
for band in E:
    if counter > n:
        break        
    plt.plot(Q, band)
    counter += 1
plt.show()
