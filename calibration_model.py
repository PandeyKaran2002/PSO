import numpy as np

class CalibrationModel:

    ''' We are implementing the calibration equation:
    B_cal = (S + Ks * T) * B_meas + (O + Ko * T)

    There are 24 parameters to fit in this equation. 

    S = [3x3], Ks = [3x3], O = [3x1], Ko = [3x1] '''

    def __init__ (self, B_meas: np.ndarray, B_ref:np.ndarray, T: np.ndarray):

        B_meas = np.asarray(B_meas)
        B_ref = np.asarray(B_ref)
        T = np.asarray(T)

        assert B_meas.ndim == 2 and B_meas.shape[1] == 3
        assert B_ref.ndim == 2 and B_ref.shape[1] == 3
        assert T.ndim == 1 and T.shape[0] == B_meas.shape[0]

        self.B_meas = B_meas
        self.B_ref = B_ref
        self.T = T
        self.N = B_meas.shape[0]

    def unpack_params(self, params: np.ndarray):

        ''' We convert the 24-length parameter vector into S (3X3), Ks (3X3), O (3X1), Ko (3X1) '''

        params = np.asarray(params).flatten()
        if params.size != 24:
            raise ValueError ("params must be length 24")
        
        S = params[0:9].reshape(3, 3)
        Ks = params[9:18].reshape(3, 3)
        O = params[18:21].reshape(3, 1)
        Ko = params[21:24].reshape(3, 1)

        return S, Ks, O, Ko
    
    def apply (self, params: np.ndarray) -> np.ndarray:

        ''' We now apply the calibration to all samples and return B_cal '''

        S, Ks, O, Ko = self.unpack_params(params)

        T_col = self.T.reshape(-1, 1)
        S_exp = S[np.newaxis, :, :]
        Ks_exp = Ks[np.newaxis, :, :]

        S_total = S_exp + Ks_exp * T_col[:, np.newaxis]

        Bm = self.B_meas.reshape(-1, 3, 1)

        B_mat = np.matmul(S_total, Bm).reshape(-1, 3)

        additive = (O.T + (Ko.T * T_col))

        B_cal = B_mat + additive

        return B_cal

    def set_dataset(self, B_meas: np.ndarray, B_ref: np.ndarray, T: np.ndarray):

        ''' Replace internal dataset for testing '''

        self.__init__(B_meas, B_ref, T)

    def residuals(self, params: np.ndarray) -> np.ndarray:

        ''' Returns residuals, i.e., B_cal - B_ref '''

        B_cal = self.apply(params)
        return B_cal - self.B_ref

    def rmse (self, params: np.ndarray) -> float:

        ''' Compute scalar RMSE over all samples and axes '''

        res = self.residuals(params)
        return np.sqrt(np.mean(np.sum(res**2, axis = 1)))


def apply_torch(self, theta, B_meas, T):
    """
    theta: [24]
    B_meas: [N, 3]
    T: [N, 1]
    """

    # Unpack parameters
    S  = theta[0:9].reshape(3, 3)
    Ks = theta[9:18].reshape(3, 3)
    O  = theta[18:21]
    Ko = theta[21:24]

    B_cal = (B_meas @ (S + Ks * T).T) + (O + Ko * T)

    return B_cal




'''
# SMOKE TEST

B_meas = np.array([[10., 0., 0.],
                   [0., 20., 0.],
                   [0., 0., 30.],
                   [5., 5., 5.]])

true_S = np.eye(3)
true_Ks = np.zeros((3,3))
true_O = np.array([[1.0], [2.0], [-1.0]])
true_Ko = np.zeros((3,1))

params = np. hstack([true_S.reshape(-1), true_Ks.reshape(-1), true_O.flatten(), true_Ko.flatten()])

T = np.array([25., 25., 25., 30.])
B_ref = ((true_S + true_Ks * 25.0) @ B_meas.T).T + (true_O.flatten() + (true_Ko.flatten() * 25.0))

cm = CalibrationModel (B_meas, B_ref, T)

B_cal = cm.apply(params)
print("B_ref:\n", B_ref)
print("B_ref:\n", B_cal)
print("RMSE:", cm.rmse(params))
'''