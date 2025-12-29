"""
This script does the following:
1. Loads real measurement data from CSV.
2. Initializes the CalibrationModel.
3. Creates the cost() wrapper.
4. Configures and runs PSO.
5. Prints and saves optimization results.

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
        method = "trf",
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

'''def plot_axis_timeseries(B_meas, B_ref, B_cal_pso, B_cal_ls):
    axes_names = ["X", "Y", "Z"]
    N = len(B_meas)
    x = np.arange(N)

    plt.figure(figsize=(15, 10))

    for i in range(3):
        plt.subplot(3, 1, i+1)

        plt.plot(x, B_ref[:, i], label="Reference", linewidth=2)
        plt.plot(x, B_meas[:, i], label="Raw", alpha=0.6)
        plt.plot(x, B_cal_pso[:, i], label="Calibrated (PSO)", alpha=0.8)
        plt.plot(x, B_cal_ls[:, i], label="Calibrated (LS)", linewidth=2)

        plt.title(f"Axis {axes_names[i]}: Raw vs Reference vs Calibrated")
        plt.xlabel("Sample Index")
        plt.ylabel("Magnetic Field (µT)")
        plt.grid(True)

        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig("After/Axis_Timeseries.png", dpi=300, bbox_inches="tight")
    plt.close()'''

'''def plot_all_in_one(B_meas, B_ref, B_cal_pso, B_cal_ls):
    axes_names = ["X", "Y", "Z"]
    N = len(B_meas)
    x = np.arange(N)

    plt.figure(figsize=(16, 10))

    for i in range(3):
        plt.subplot(3, 1, i+1)

        plt.plot(x, B_meas[:, i], label="Raw", alpha=0.5)
        plt.plot(x, B_ref[:, i], label="Reference", linewidth=2)
        plt.plot(x, B_cal_pso[:, i], label="Calibrated (PSO)", alpha=0.8)
        plt.plot(x, B_cal_ls[:, i], label="Calibrated (LS)", linewidth=2)

        plt.title(f"Axis {axes_names[i]} — Raw vs Ref vs PSO vs LS")
        plt.xlabel("Sample Index")
        plt.ylabel("Magnetic Field (µT)")
        plt.grid(True)

        if i == 0:
            plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("After/All_In_One.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_combined_all_axes(B_meas, B_ref, B_cal_pso, B_cal_ls):
    x = np.arange(len(B_meas))

    # Colors for X, Y, Z
    axis_colors = ["red", "green", "blue"]
    axis_names = ["X", "Y", "Z"]

    plt.figure(figsize=(15, 6))

    # ---- Raw ----
    for i in range(3):
        plt.plot(x, B_meas[:, i], 
                 color=axis_colors[i], 
                 alpha=0.3,
                 label=f"Raw {axis_names[i]}" if i == 0 else None)

    # ---- Reference ----
    for i in range(3):
        plt.plot(x, B_ref[:, i], 
                 color=axis_colors[i], 
                 linestyle="--",
                 alpha=0.7,
                 label=f"Reference {axis_names[i]}" if i == 0 else None)

    # ---- PSO Calibrated ----
    for i in range(3):
        plt.plot(x, B_cal_pso[:, i], 
                 color=axis_colors[i], 
                 linestyle="-.",
                 alpha=0.8,
                 label=f"PSO Cal {axis_names[i]}" if i == 0 else None)

    # ---- LS Calibrated ----
    for i in range(3):
        plt.plot(x, B_cal_ls[:, i], 
                 color=axis_colors[i], 
                 linewidth=2,
                 label=f"LS Cal {axis_names[i]}" if i == 0 else None)

    plt.title("Magnetic Field — All Axes & All Calibration Stages")
    plt.xlabel("Sample Index")
    plt.ylabel("Magnetic Field (µT)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("After/Combined_All_Axis.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_four_panel_summary(B_meas, B_ref, B_cal_pso, B_cal_ls):
    axes_names = ["X", "Y", "Z"]
    n = len(B_meas)
    idx = np.arange(n)

    # Residuals
    res_raw = B_meas - B_ref
    res_pso = B_cal_pso - B_ref
    res_ls  = B_cal_ls  - B_ref

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Magnetometer Calibration Summary — Raw vs PSO vs LS", fontsize=16)

    # ------------------------------------------------------------
    # Panel A — Raw vs Reference
    # ------------------------------------------------------------
    ax = axs[0, 0]
    for i, col in enumerate(axes_names):
        ax.plot(idx, B_meas[:, i], alpha=0.5, label=f"Raw {col}")
        ax.plot(idx, B_ref[:, i], linestyle='--', label=f"Reference {col}")
    ax.set_title("A. Raw Measurements vs Reference")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Magnetic Field (µT)")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)

    # ------------------------------------------------------------
    # Panel B — PSO-Calibrated vs Reference
    # ------------------------------------------------------------
    ax = axs[0, 1]
    for i, col in enumerate(axes_names):
        ax.plot(idx, B_cal_pso[:, i], alpha=0.8, label=f"PSO Cal {col}")
        ax.plot(idx, B_ref[:, i], linestyle='--', label=f"Reference {col}")
    ax.set_title("B. PSO-Calibrated vs Reference")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Magnetic Field (µT)")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)

    # ------------------------------------------------------------
    # Panel C — LS-Calibrated vs Reference
    # ------------------------------------------------------------
    ax = axs[1, 0]
    for i, col in enumerate(axes_names):
        ax.plot(idx, B_cal_ls[:, i], alpha=0.8, label=f"LS Cal {col}")
        ax.plot(idx, B_ref[:, i], linestyle='--', label=f"Reference {col}")
    ax.set_title("C. LS-Refined vs Reference")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Magnetic Field (µT)")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)

    # ------------------------------------------------------------
    # Panel D — Residuals Comparison
    # ------------------------------------------------------------
    ax = axs[1, 1]
    for i, col in enumerate(axes_names):
        ax.plot(idx, res_raw[:, i], alpha=0.4, label=f"Raw Residual {col}")
        ax.plot(idx, res_pso[:, i], alpha=0.7, label=f"PSO Residual {col}")
        ax.plot(idx, res_ls[:, i],  alpha=1.0, label=f"LS Residual {col}")
    ax.set_title("D. Residuals (Error vs Reference)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Error (µT)")
    ax.grid(True)
    ax.legend(ncol=3, fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("After/Four_Panel_Summary.png", dpi=300, bbox_inches="tight")
    plt.close()'''



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

    '''
    We will plot the results before Least Squares is implemented
    '''

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

'''
    # RMSE vs Iteration

    plt.figure(figsize=(8,4))
    plt.plot(swarm.history, label="Global Best RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE (µT)")
    plt.title("PSO Convergence")
    plt.grid(True)
    plt.legend()
    plt.savefig("Before/RMSE vs Iteration.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Particle Position Projections in 2D

    plot1  = plot_particles_2d(swarm, 0, 1)
    plot1.savefig("Before/Particle Position Projections 01.png", dpi=300, bbox_inches="tight")
    plot1.close()
    plot2 = plot_particles_2d(swarm, 4, 5)
    plot2.savefig("Before/Particle Position Projections 45.png", dpi=300, bbox_inches="tight")
    plot2.close()
    plot3 = plot_particles_2d(swarm, 8, 9)
    plot3.savefig("Before/Particle Position Projections 89.png", dpi=300, bbox_inches="tight")
    plot3.close()

    # Parameter Convergence Plot

    param_hist = np.array(swarm.param_history)  # shape = (iters, 24)

    plt.figure(figsize=(12,6))
    for i in range(param_hist.shape[1]):
        plt.plot(param_hist[:, i], label=f"p{i}", alpha=0.7)

    plt.title("Parameter Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.grid(True)
    plt.legend(ncol=4, fontsize=6)
    plt.savefig("Before/Parameter Convergence Plot Before LS.png", dpi=300, bbox_inches="tight")
    plt.close()


    #   Per-Axis Calibration Error Visualization  #


    # Compute calibrated B-field
    model = get_model()
    B_cal = model.apply(best_pos)

    # Residuals: difference between calibrated and reference
    res = B_cal - B_ref

    plt.figure(figsize=(12, 4))
    axes = ["X", "Y", "Z"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(res[:, i], bins=50, alpha=0.7)
        plt.title(f"Residuals: Axis {axes[i]}")
        plt.xlabel("Error (µT)")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("Before/Residuals Before LS.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3D Ellipsoid Before/After Calibration

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(B_meas[:,0], B_meas[:,1], B_meas[:,2], s=1, alpha=0.2, label="Raw", color = "blue")
    ax.scatter(B_cal[:,0],  B_cal[:,1],  B_cal[:,2],  s=1, alpha=0.2, label="Calibrated", color = "crimson")
    ax.scatter(B_ref[:,0],  B_ref[:,1],  B_ref[:,2],  s=1, alpha=0.2, label="Reference", color = "springgreen")

    ax.set_title("Magnetic Field Vectors in 3D Space")
    ax.legend(markerscale=2, fontsize=12, frameon=True)
    for lh in ax.legend().legendHandles:
        lh.set_alpha(1)
    ax.set_xlabel("Magnetic Field X (µT)")
    ax.set_ylabel("Magnetic Field Y (µT)")
    ax.set_zlabel("Magnetic Field Z (µT)")
    plt.savefig("Before/3D Ellipsoid Before LS.png", dpi=300, bbox_inches="tight")
    plt.close()


    # Velocity Evaluation

    plt.plot(swarm.velocity_history)
    plt.title("Average Particle Velocity per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Velocity Magnitude")
    plt.grid(True)
    plt.savefig("Before/Velocity Evaluation Before LS.png", dpi=300, bbox_inches="tight")
    plt.close()'''


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

    '''print("\nGenerating plots using refined parameters after Least Squares...")'''

    print("\n===== COMPUTATION TIME SUMMARY =====")
    print(f"PSO time: {pso_time:.3f} seconds")
    print(f"LS time: {ls_time:.3f} seconds")
    print(f"Total PSO + LS time: {(pso_time + ls_time):.3f} seconds")
    print("===================================\n")

    model = get_model()
    B_cal_refined = model.apply(refined_params)

    # Residual Histogram After LS

    res_refined = B_cal_refined - B_ref

    '''plt.figure(figsize = (12, 4))
    axes = ["X", "Y", "Z"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(res_refined[:, i], bins = 50, alpha = 0.7, color = "green")
        plt.title(f"Residuals After LS: Axis {axes[i]}")
        plt.xlabel("Error (µT)")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("After/Residual Histogram After LS.png", dpi=300, bbox_inches="tight")
    plt.close()


    # 3D Ellipsoid Before Vs After LS

    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111, projection = "3d")

    ax.scatter(B_meas[:, 0], B_meas[:,1], B_meas[:,2], s = 1, alpha = 0.15, label = "Raw", color = "blue")
    ax.scatter(B_cal_refined[:, 0], B_cal_refined[:,1], B_cal_refined[:,2], 
                s = 1, alpha = 0.2, label = "Calibrated (LS)", color = "crimson")
    ax.scatter(B_ref[:,0], B_ref[:,1], B_ref[:,2], s = 1, alpha = 0.2, label = "Reference", color = "springgreen")

    ax.set_title("Magnetic Field Vectors (Refined LS Calibration)")
    ax.legend(markerscale=2, fontsize=12, frameon=True)
    for lh in ax.legend().legendHandles:
        lh.set_alpha(1)
    ax.set_xlabel("Magnetic Field X (µT)")
    ax.set_ylabel("Magnetic Field Y (µT)")
    ax.set_zlabel("Magnetic Field Z (µT)")
    plt.savefig("After/3D Ellipsoid After LS.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Axis Scatter Before vs After

    plt.figure(figsize = (12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.scatter(B_ref[:, i], B_cal_refined[:, i], s = 5, alpha  = 0.5,
                    color = "darkorange")
        plt.xlabel("Reference (µT)")
        plt.ylabel("Calibrated (µT)")
        plt.title(f"Axis {axes[i]} After LS")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("After/Axis Scatter After LS.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Line Graph

    plot_axis_timeseries(B_meas, B_ref, B_cal, B_cal_refined)
    plot_all_in_one(B_meas, B_ref, B_cal, B_cal_refined)
    plot_combined_all_axes(B_meas, B_ref, B_cal, B_cal_refined)
    plot_four_panel_summary(B_meas, B_ref, B_cal, B_cal_refined)'''




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


    '''os.makedirs("results", exist_ok = True)
    save_path = "results/calibration_params.json"

    with open(save_path, "w") as f:
        json.dump(results, f, indent = 4)

    print(f"\nSaved calibrated parameters to: {save_path}")'''


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

'''def plot_particles_2d(swarm, dim_x=0, dim_y=1):
    xs = [p.position[dim_x] for p in swarm.particles]
    ys = [p.position[dim_y] for p in swarm.particles]

    plt.figure(figsize=(5,5))
    plt.scatter(xs, ys, alpha=0.6)
    plt.xlabel(f"param {dim_x}")
    plt.ylabel(f"param {dim_y}")
    plt.title("Particle distribution in 2D parameter space")
    plt.grid(True)
    return plt
    #plt.show()'''



if (__name__ == "__main__"):
    run_pso("full_data.csv")


