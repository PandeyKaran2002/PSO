import numpy as np
from typing import Optional
from calibration_model import CalibrationModel

_model: Optional[CalibrationModel] = None
_REG_L2_WEIGHT = 0.0     # Useful if we want to add a penalty later

def init_cost_model(B_meas: np.ndarray, B_ref: np.ndarray, T: np.ndarray):

    ''' We initialise the CalibrationModel used by cost(params). 
    We will only call it once before calling cost(params). '''

    global _model
    _model = CalibrationModel(B_meas, B_ref, T)

def _ensure_model():

    if _model is None:
        raise Runtimerror("Cost model not initialised. Call _init_cost_model(B_meas, B_ref, T) first")

def get_model() -> CalibrationModel:
    _ensure_model()
    return _model

def _l2_regularisation(params: np.ndarray) -> float:

    ''' Optional '''

    if _REG_L2_WEIGHT == 0.0:
        return 0.0

    params = np.asarray(params).ravel()
    return _REG_L2_WEIGHT * np.sum(params ** 2)

def cost(params: np.ndarray) -> float:

    '''
    Compute the RMSE for a given parameter vector (length 24).

    This is the main function PSO will call:
        cost_value = cost(params)

    Steps:
    1. Ensure the model has been initialized
    2. Convert params to a clean (24,) vector
    3. Apply calibration model → B_cal
    4. Compute residuals = B_cal - B_ref
    5. Compute RMSE
    6. Add optional regularization
    '''

    _ensure_model()

    params = np.asarray(params).ravel()
    if params.size != 24:
        raise ValueError(f"params must be length 24, got {params.size}")
    
    #Applying the calibration equation
    B_cal = _model.apply(params)

    residuals = B_cal - _model.B_ref
    mse = np.mean(np.sum(residuals **2, axis = 1))
    rmse = float(np.sqrt(mse))

    rmse += _l2_regularisation(params)     # Optional

    return rmse

def cost_batch(params_batch: np.ndarray) -> np.ndarray:

    '''
    This functions evaluates the cost for multiple parameter vectors at once.
    - params_batch: shape (n_candidates, 24)
    - returns: array of RMSE values of shape (n_candidates,)

    This is optional, but useful if the PSO implementation wants to evaluate
    the entire swarm in one call. 
    '''

    _ensure_model()

    X = np.asarray(params_batch)

    #Case 1: a single vector is passed -> return an array with one value

    if X.ndim ==1:
        return np.array([cost(X)])

    #Case 2: batch must be (N, 24)

    if X.shape[1] != 24:
        raise ValueError(f"params_batch must have shape (n, 24), got {X.shape}")
    
    n = X.shape[0]
    costs = np.zeros(n)

    #Loop through candidates
    for i in range(n):
        costs[i] = cost(X[i])

    return costs


'''
B_meas = np.array([
    [10.,  0.,  0.],
    [ 0., 20.,  0.],
    [ 0.,  0., 30.],
    [ 5.,  5.,  5.]
])

# Temperature values for each measurement
T = np.array([25., 25., 25., 30.])

# True calibration parameters (ground truth)
# Identity scale matrix S
S_true = np.eye(3)

# Zero temperature matrix Ks
Ks_true = np.zeros((3, 3))

# Bias vector O
O_true = np.array([[1.0], [2.0], [-1.0]])

# Zero temperature offset Ko
Ko_true = np.zeros((3, 1))

# Pack them into a parameter vector of length 24
params_true = np.hstack([
    S_true.reshape(-1),
    Ks_true.reshape(-1),
    O_true.flatten(),
    Ko_true.flatten()
])

# -------------------------------------------------
# 2. Compute the reference magnetic field B_ref
# -------------------------------------------------

# (S + Ks*T) * B_meas  +  (O + Ko*T)

# Compute B_ref correctly using per-sample matrix multiplication
B_ref = np.zeros_like(B_meas)

for i in range(len(T)):
    S_i = S_true + Ks_true * T[i]     # (3,3)
    O_i = O_true.flatten() + Ko_true.flatten() * T[i]   # (3,)
    B_ref[i] = S_i @ B_meas[i] + O_i


# -------------------------------------------------
# 3. Initialize the cost model
# -------------------------------------------------

init_cost_model(B_meas, B_ref, T)

# -------------------------------------------------
# 4. Evaluate cost on the true parameters
# -------------------------------------------------

print("Testing cost(params_true)...")
val = cost(params_true)
print("Cost =", val)

# Should be extremely close to 0 (perfect match)
assert abs(val) < 1e-12, "Cost should be zero for true parameters!"

# -------------------------------------------------
# 5. Test cost_batch()
# -------------------------------------------------

print("\nTesting cost_batch...")

# A wrong parameter vector (slightly changed S11)
params_bad = params_true.copy()
params_bad[0] = 0.9

batch = np.vstack([params_true, params_bad])
vals = cost_batch(batch)

print("Batch costs =", vals)

assert vals[0] < 1e-12, "First cost in batch should be 0."
assert vals[1] > 0.0, "Second cost in batch should be > 0 (it is wrong)."

print("\nAll cost_function.py tests passed.")'''