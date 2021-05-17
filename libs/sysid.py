import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd


class sysid(object):

    def regMatrix_ARX(self, y, u, na, nb, p):
        N = y.size

        if N != u.size:
            exit(f"input-output vectors should have the same size: \
                 in: {y.size} and out: {u.size}")

        Phi = np.zeros((N-p, na+nb))
        colPhi = []
        for i in range(0, na):
            Phi[:, i] = -y[(p-i-1):(N-i-1)]
            colPhi.append(f"-y(k-{i+1})")
        for i in range(0, nb):
            Phi[:, na+i] = u[(p-i-1):(N-i-1)]
            colPhi.append(f"u(k-{i+1})")

        rowPhi = [f"k={x}" for x in range(p, N)]
        ret = pd.DataFrame(Phi, columns=colPhi, index=rowPhi)
        return ret

    def regMatrix_MA(self, y, u, e, na, nb, nc, p):
        # creates the regression matrix
        N = y.size

        if (N != u.size or N != e.size):
            exit(f"input vectors should have the same size \
                 in: {y.size}, out: {u.size} and e: {e.size}")

        Phi = np.zeros((N-p, na+nb+nc))
        colPhi = []
        for i in range(0, na):
            Phi[:, i] = -y[(p-i-1):(N-i-1)]
            colPhi.append(f"-y(k-{i+1})")
        for i in range(0, nb):
            Phi[:, na+i] = u[(p-i-1):(N-i-1)]
            colPhi.append(f"u(k-{i+1})")
        for i in range(0, nc):
            Phi[:, na+nb+i] = e[(p-i-1):(N-i-1)]
            colPhi.append(f"e(k-{i+1})")

        rowPhi = [f"k={x}" for x in range(p, N)]

        ret = pd.DataFrame(Phi, columns=colPhi, index=rowPhi)
        return ret

    def targetVec(self, y, p):
        N = y.size
        Y = np.array([y[p:]]).T
        ret = pd.DataFrame(Y, columns=["y(k)"],
                           index=[f"k={x}" for x in range(p, N)])
        return ret

    def calcFR_ARX(self, y, u, na, nb, p, th_hat):
        y_fr = y[:p]
        u_fr = u
        N = y.size

        for k in range(p, N):
            phi_k = self.regMatrix_ARX(np.append(y_fr[(k-p):(k)], 0),
                                       np.append(u_fr[(k-p):(k)], 0),
                                       na, nb, p)
            mat = np.matmul(phi_k.to_numpy(), th_hat)
            y_fr = np.append(y_fr, mat)

        y_fr = y_fr[p:N]
        return y_fr

    def calcOSA_ARMAX(self, y, u, na, nb, nc, p, th_hat):
        y_fr = y[:p]
        u_fr = u
        e_fr = np.zeros(p)
        N = y.size

        for k in range(p, N):
            auxy = np.append(y_fr[(k-p):(k)], 0)
            auxu = np.append(u_fr[(k-p):(k)], 0)
            auxe = np.append(e_fr[(k-p):(k)], 0)
            phi_k = self.regMatrix_MA(auxy, auxu, auxe, na, nb, nc, p)
            y_fr = np.append(y_fr, np.matmul(phi_k, th_hat))
            e_fr = np.append(e_fr, y[k] - y_fr[k])

        y_fr = y_fr[p:N]

        return y_fr

    def db(self, X):
        pass

    def randnoise(self, N, cutoff):
        # generate low-pass filtered random noise
        # N number of samples
        # cutoff cutoff normalized frequency
        # adapted from Pintelon,Schoukens book

        b, a = signal.butter(6, cutoff, btype='low', output='ba')
        noise = sp.signal \
            .filtfilt(b, a, np.random.normal(loc=0, scale=1, size=N))
        return noise

    def multisine(self, N, cutoff):
        # generate low-pass filtered random noise
        # N number of samples
        # cutoff cutoff normalized frequency
        # adapted from Pintelon, Schoukens book

        # fSample = 1  # use normalized frequency  # not used
        # Ts = 1/fSample  # not used
        NSines = round(N*cutoff)
        # f = np.arange(N) * fSample / N  # not used

        U = np.zeros((N, 1), dtype=complex)
        U[1:(NSines+1)] = np.exp(1j*2*np.pi*np.random.rand(NSines))\
                            .reshape((NSines, 1))
        u = 2*np.real(np.fft.ifft(U.reshape((N,))))
        u = u/np.std(u)
        return u

    def M_spec(self, u, title='u'):
        pass

    # Calculates R2
    def calcR2(self, real, est):
        SSE = np.sum((np.subtract(real, est))**2)
        avg_real = np.average(real)
        sum2 = sum((np.subtract(real, avg_real))**2)
        R2 = 1 - (SSE / sum2)
        return R2

    def CGS(self, P):
        pass

    def MGS(self, P):
        # Modified Gram-Schimidt factorization
        # Aguirre 2015 book
        # obtains P = Q * A
        # where P is N x Nth
        # so that
        # Q is a N x Nth matrix with orthogonal columns
        # A is a Nth x Nth unit upper triangular matrix

        N = P.shape[0]
        Nth = P.shape[1]

        # init matrix
        A = np.diag([1.0 for x in range(Nth)])
        P_i_1 = P
        Q = np.zeros((N, Nth))
        P_i = np.zeros((N, Nth))
        for i in range(Nth-1):
            Q[:, i] = P_i_1[:, i].reshape(P_i_1[:, i].shape[0])
            for j in range(i+1, Nth):
                # disp(j,i)
                A[i, j] = float(np.divide(np.matmul(Q[:, i], P_i_1[:, j]),
                                np.matmul(Q[:, i], Q[:, i])))
                P_i[:, j] = (P_i_1[:, j].reshape((P_i_1[:, j].shape[0],)) -
                             (A[i, j] * Q[:, i]))
            P_i_1 = P_i
        # THE END
        Q[:, Nth-1] = P_i_1[:, Nth-1]

        return (A, Q)

    def regMatNARX(self, u, y, nu, ny, L):
        # n = nu + ny
        # p = max(nu, ny) + 1
        # auxexp = []
        # candlist = []
        # # generate all terms product combinations possible
        # for i in range(L):
        #     # generate input args for expand.grid
        #     eval(parse(text=paste0("auxexp$x",i,"=1:n")))
        #     # call expand.grid for changing number of arguments
        #     cand = do.call(expand.grid,auxexp)

        #     # order each row of the matrix
        #     cand = t(apply(cand,1,sort))
        #     # keep unique rows
        #     cand = unique(cand)
        #     candlist[[i]] = cand
        pass

    def frols(self, P, Y, rho):
        pass

    def regMatNARMAX(self, u, y, e, nu, ny, ne, p, L, selectTerms):
        pass


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = np.array([2, 4, 6])
    print(sysid().calcR2(a, b))
