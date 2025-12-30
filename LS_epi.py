"""
This script implements PSO+LS, LS+PSO, and LS-alone models. Also provides epistemic uncertainties.

"""

import numpy as np
import pandas as pd 
import json
import os 
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

from calibration_model import CalibrationModel
from cost_function import init_cost_model, cost
from pso import Swarm
from cost_function import get_model

def load_data (csv_path: str):

    '''
    Columns are:
    0: Time
    1: refX
    2: refY
    3: refZ
    4: testX
    5: testY
    6: testZ
    7: Temp
    '''

    df = pd.read_csv(csv_path, header = None)

    df.columns = ["Time", "refX", "refY", "refZ", "testX", "testY", "testZ", "Temp"]

    df["Temp"] = df["Temp"] - 273.15

    B_meas = df[["testX", "testY", "testZ"]].to_numpy()
    B_ref = df[["refX", "refY", "refZ"]].to_numpy()
    T = df["Temp"].to_numpy()

    import numpy as np
    print("N rows:", len(df))
    print("Temp (C) min,max:", df["Temp"].min(), df["Temp"].max())
    print("ref ranges:", np.min(B_ref, axis=0), np.max(B_ref, axis=0))
    print("meas ranges:", np.min(B_meas, axis=0), np.max(B_meas, axis=0))


    return B_meas, B_ref, T

def initialise_model(csv_path):
    '''
    Initialise the CalibrationModel and connect it to the cost() function.
    '''

    B_meas, B_ref, T = load_data(csv_path)
    print("Number of samples: ", len(B_meas))

    # Initialise the global model inside cost_function.py

    init_cost_model(B_meas, B_ref, T)

    print ("Calibration model initialised.")

    return B_meas, B_ref, T

def make_bounds():
    # Returns (lower_bounds, upper_bounds) for the 24 parameters.

    S_diag_low = 0.5
    S_diag_high = 1.5
    S_off_low = -0.5
    S_off_high = 0.5

    lower_S = np.array([
        S_diag_low, S_off_low, S_off_low,
        S_off_low, S_diag_low, S_off_low,
        S_off_low, S_off_low, S_diag_low,
        ])

    upper_S = np.array([
        S_diag_high, S_off_high, S_off_high,
        S_off_high, S_diag_high, S_off_high,
        S_off_high, S_off_high, S_diag_high,
        ])

    lower_Ks = -0.05 * np.ones(9)
    upper_Ks = 0.05 * np.ones(9)

    lower_O = -1.0 * np.ones(3)
    upper_O = 1.0 * np.ones(3)

    lower_Ko = -0.05 * np.ones(3)
    upper_Ko = 0.05 * np.ones(3)

    # Concatenate all

    lower_bounds = np.concatenate([lower_S, lower_Ks, lower_O, lower_Ko])
    upper_bounds = np.concatenate([upper_S, upper_Ks, upper_O, upper_Ko])

    return lower_bounds, upper_bounds


def initial_guess_identity():
    """
    Physically neutral initial guess:
    S = I, Ks = 0, O = 0, Ko = 0
    """
    S0 = np.eye(3)
    Ks0 = np.zeros((3, 3))
    O0 = np.zeros(3)
    Ko0 = np.zeros(3)

    return np.hstack([
        S0.flatten(),
        Ks0.flatten(),
        O0,
        Ko0
    ])


def ls_residuals(params, B_meas, B_ref, T):
    
    # Residual Vector for LS refinement
    

    model = get_model()
    B_cal_ls = model.apply(params)

    return (B_cal_ls - B_ref).flatten()

def refine_least_squares(initial_params, B_meas, B_ref, T):

    result = least_squares(
        fun = ls_residuals,
        x0 = initial_params,
        args = (B_meas, B_ref, T),
        method = "lm",
        max_nfev = 10000,
        xtol = 1e-12,
        ftol = 1e-12,
        gtol = 1e-12
    )

    refined_params = result.x 
    residual_vec = ls_residuals(refined_params, B_meas, B_ref, T)
    refined_rmse = np.sqrt(np.mean(residual_vec**2))

    print("Refined RMSE: ", refined_rmse)

    return refined_params, refined_rmse


def run_pso(csv_path):
    # Full Pipeline

    B_meas, B_ref, T = initialise_model(csv_path)

    print("\nModel and data ready.")
    print("Running PSO on", len(B_meas), "samples...")

    lower_bounds, upper_bounds = make_bounds()
    bounds_main = (lower_bounds, upper_bounds)

    swarm = Swarm(n_particles = 200, dim = 24, bounds  = bounds_main, inertia = 0.5, cognitive = 1.5, social = 1.5)

    print("Swarm Initialised With 200 Particles.")

    t_pso_start = time.perf_counter()

    best_pos, best_cost = swarm.run(n_iterations = 200, verbose = True)

    t_pso_end = time.perf_counter()
    pso_time = t_pso_end - t_pso_start

    print(f"\nPSO computation time: {pso_time:.3f} seconds")

    print("\n===== PSO Finished ====")
    print("Best RMSE:", best_cost)
    print("Best Parameters:")
    print(best_pos)


    S = best_pos[0:9].reshape(3, 3)
    Ks = best_pos[9:18].reshape(3, 3)
    O = best_pos[18:21].reshape(3)
    Ko = best_pos[21:24].reshape(3)

    print("\n====================")
    print("  CALIBRATION MATRICES Before LS")
    print("====================\n")

    print("S (Soft-iron 3×3):")
    print(S, "\n")

    print("Ks (Temp soft-iron 3×3):")
    print(Ks, "\n")

    print("O (Hard-iron offset 3×1):")
    print(O, "\n")

    print("Ko (Temp offset 3×1):")
    print(Ko, "\n")


    # ---- Least Squares Refinement ----

    t_ls_start = time.perf_counter()

    refined_params, refined_rmse = refine_least_squares(best_pos, B_meas, B_ref, T)

    t_ls_end = time.perf_counter()
    ls_time = t_ls_end - t_ls_start

    print(f"Least Squares computation time: {ls_time:.3f} seconds")

    print("\n===== After Least-Squares Refinement =====")
    print("Refined RMSE:", refined_rmse)
    print("Refined Parameters:")
    print(refined_params)

    print("\n===== COMPUTATION TIME SUMMARY =====")
    print(f"PSO time: {pso_time:.3f} seconds")
    print(f"LS time: {ls_time:.3f} seconds")
    print(f"Total PSO + LS time: {(pso_time + ls_time):.3f} seconds")
    print("===================================\n")

    model = get_model()
    B_cal_refined = model.apply(refined_params)

    # RMSE Summary

    print("\n===== RMSE Summary =====")
    print("PSO RMSE:", best_cost)
    print("Refined LS RMSE:", refined_rmse)
    print("Improvement:", best_cost - refined_rmse)
    print("========================\n")


    # results = {"best_cost": float(best_cost), "best_params": best_pos.tolist(),}

    results = {
    "pso_best_cost": float(best_cost),
    "pso_best_params": best_pos.tolist(),
    "refined_rmse": float(refined_rmse),
    "refined_params": refined_params.tolist(),
    }


    # Parameters After Refinement

    S2 = refined_params[0:9].reshape(3, 3)
    Ks2 = refined_params[9:18].reshape(3, 3)
    O2 = refined_params[18:21].reshape(3)
    Ko2 = refined_params[21:24].reshape(3)

    print("\n====================")
    print("  CALIBRATION MATRICES After LS")
    print("====================\n")

    print("S (Soft-iron 3×3):")
    print(S2, "\n")

    print("Ks (Temp soft-iron 3×3):")
    print(Ks2, "\n")

    print("O (Hard-iron offset 3×1):")
    print(O2, "\n")

    print("Ko (Temp offset 3×1):")
    print(Ko2, "\n")




    return best_pos, best_cost, refined_params, refined_rmse

def run_ls_then_pso(csv_path, n_trials=5):
    """
    LS → PSO diagnostic experiment.
    Runs LS-first multiple times to test stability,
    then runs PSO starting from LS solution.
    """

    B_meas, B_ref, T = initialise_model(csv_path)
    lower_bounds, upper_bounds = make_bounds()
    bounds = (lower_bounds, upper_bounds)

    final_rmses = []
    final_params = []

    print("\n==============================")
    print(" Running LS → PSO experiment ")
    print("==============================\n")

    for trial in range(n_trials):
        print(f"\n--- Trial {trial+1}/{n_trials} ---")

        # ---- LS FIRST ----
        init_params = initial_guess_identity()
        init_params += 0.01 * np.random.randn(24)  # small perturbation

        ls_params, ls_rmse = refine_least_squares(
            init_params, B_meas, B_ref, T
        )

        print("LS RMSE:", ls_rmse)
        print(ls_params)

        # ---- PSO AFTER LS ----
        swarm = Swarm(
            n_particles=50,
            dim=24,
            bounds=bounds,
            inertia=0.5,
            cognitive=1.5,
            social=1.5
        )

        # Initialize swarm around LS solution
        spread = 0.1 * (upper_bounds - lower_bounds)

        for p in swarm.particles:
            p.position = ls_params + np.random.uniform(-spread, spread)
            p.position = np.clip(p.position, lower_bounds, upper_bounds)
            p.velocity = np.zeros(24)

            p.best_position = p.position.copy()
            p.best_cost = cost(p.position)

        swarm.global_best_position = ls_params.copy()
        swarm.global_best_cost = cost(ls_params)

        best_pos, best_cost = swarm.run(n_iterations=150, verbose=False)

        print("Final LS → PSO RMSE:", best_cost)

        final_rmses.append(best_cost)
        final_params.append(best_pos)

    final_rmses = np.array(final_rmses)
    final_params = np.array(final_params)

    print("\n==============================")
    print(" LS → PSO SUMMARY ")
    print("==============================")
    print("Mean RMSE:", final_rmses.mean())
    print("Std RMSE :", final_rmses.std())

    compute_epistemic_uncertainty(final_params, name="(LS → PSO)")


    # Parameter stability (soft-iron diagonals)
    S_diag = final_params[:, [0, 4, 8]]
    print("\nSoft-iron diagonal std dev:")
    print("Sxx:", S_diag[:,0].std())
    print("Syy:", S_diag[:,1].std())
    print("Szz:", S_diag[:,2].std())

    return final_rmses, final_params

def run_pso_then_ls_multiple(csv_path, n_trials=5):
    """
    PSO → LS stability benchmark
    """

    B_meas, B_ref, T = initialise_model(csv_path)
    lower_bounds, upper_bounds = make_bounds()
    bounds = (lower_bounds, upper_bounds)

    rmses = []
    params = []

    for trial in range(n_trials):
        print(f"\n--- PSO → LS Trial {trial+1}/{n_trials} ---")

        swarm = Swarm(
            n_particles=100,
            dim=24,
            bounds=bounds,
            inertia=0.5,
            cognitive=1.5,
            social=1.5
        )

        best_pos, best_cost = swarm.run(n_iterations=300, verbose=False)
        refined_params, refined_rmse = refine_least_squares(
            best_pos, B_meas, B_ref, T
        )

        rmses.append(refined_rmse)
        params.append(refined_params)

        print("Final RMSE:", refined_rmse)

    rmses = np.array(rmses)
    params = np.array(params)

    print("\n==============================")
    print(" PSO → LS SUMMARY ")
    print("==============================")
    print("Mean RMSE:", rmses.mean())
    print("Std RMSE :", rmses.std())

    compute_epistemic_uncertainty(params,name="(PSO → LS)")


    S_diag = params[:, [0, 4, 8]]
    print("\nSoft-iron diagonal std dev:")
    print("Sxx:", S_diag[:,0].std())
    print("Syy:", S_diag[:,1].std())
    print("Szz:", S_diag[:,2].std())

    return rmses, params

def compute_epistemic_uncertainty(params, name=""):
    """
    params: array of shape (n_trials, 24)
    """
    params = np.asarray(params)

    param_std = params.std(axis=0)

    print(f"\n===== Epistemic Uncertainty {name} =====")
    print("Parameter-wise std:")
    print(param_std)

    # Soft-iron diagonals
    Sxx_std = param_std[0]
    Syy_std = param_std[4]
    Szz_std = param_std[8]

    print("\nSoft-iron diagonal epistemic std:")
    print("Sxx:", Sxx_std)
    print("Syy:", Syy_std)
    print("Szz:", Szz_std)

    return param_std

def run_ls_only_multiple(csv_path, n_trials=20):
    B_meas, B_ref, T = initialise_model(csv_path)

    params = []
    rmses = []

    for i in range(n_trials):
        init_params = initial_guess_identity()
        init_params += 0.01 * np.random.randn(24)

        p, rmse = refine_least_squares(
            init_params, B_meas, B_ref, T
        )

        params.append(p)
        rmses.append(rmse)

    params = np.array(params)
    rmses = np.array(rmses)

    print("\n==============================")
    print(" LS-ONLY SUMMARY ")
    print("==============================")
    print("Mean RMSE:", rmses.mean())
    print("Std RMSE :", rmses.std())

    compute_epistemic_uncertainty(
        params,
        name="(LS only)"
    )

    return rmses, params


if (__name__ == "__main__"):
    print("\n===== PSO → LS =====")
    run_pso_then_ls_multiple("full_data.csv")

    print("\n===== LS → PSO =====")
    run_ls_then_pso("full_data.csv")

    print("\n===== LS ONLY =====")
    run_ls_only_multiple("full_data.csv")



