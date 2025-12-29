'''
PSO implementation for magnetometer calibration. 
This module defines the Particle and Swarm classes. 
The Swarm uses cost_function.cost(params) to evaluate fitness.
'''

import numpy as np
from typing import Optional
from cost_function import cost

class Particle:

    '''

    This represents a single particle in the PSO Swarm.

    Each particle has:

    - position = Current parameter vector (size = 24)
    - velocity = how the particle moves in parameter space
    - best_position = best position this particle has discovered
    - best_cost = corresponding cost of best_position

    '''

    def __init__(self, dim: int, bounds: tuple[np.ndarray, np.ndarray]):
        '''

        Initialise a particle.

        dim: number of particles (24)
        bounds: (lower_bounds, upper_bounds)
                each is shape (dim,)
        
        This will create:
        - random initial position within bounds
        - random initial velocity (small)
        
        '''
        self.dim = dim

        self.lower_bounds, self.upper_bounds = bounds

        self.position = np.random.uniform(self.lower_bounds, self.upper_bounds)

        vel_range = (self.upper_bounds - self.lower_bounds) * 0.1
        self.velocity = np.random.uniform(-vel_range, vel_range)
        
        self.velocity_history = []

        current_cost = cost(self.position)

        self.best_position = self.position.copy()
        self.best_cost = current_cost

    def update_velocity(self, global_best_position: np.ndarray, inertia: float, cognitive: float, social: float,):

        '''
        We update the velocity using the formula:

        v = w*v + c1*r1*(personal_best - position) + c2*r2*(global_best - position)

        - inertia: w (controls momentum)
        - cognitive: c1 (pull toward personal best)
        - social: c2 (pull toward global best)
        '''

        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)

        cognitive_term = cognitive*r1*(self.best_position - self.position)
        social_term = social*r2*(global_best_position - self.position)

        self.velocity = (inertia*self.velocity + cognitive_term + social_term)

    def update_position(self):

        '''
        Update particle position:
        position = position + velocity
        Then clamp the position within bounds.
        '''
    
        self.position = self.position + self.velocity

        self.position = np.clip(self.position, self.lower_bounds, self.upper_bounds)

    def evaluate(self):

        '''
        Evaluate the particle at its current position and update its personal best if improved. '''

        curr_cost = cost(self.position)

        if (curr_cost < self.best_cost):
            self.best_cost = curr_cost
            self.best_position = self.position.copy()



class Swarm:
    '''
    Swarm containing multiple particles and running the PSO algorithm
    '''

    def __init__(self, n_particles: int, dim: int, bounds: tuple[np.ndarray, np.array], inertia: float = 0.5, cognitive: float = 1.5, social: float = 1.5,):
        '''
        Initialising the Swarm.

        Parameters:
        - n_particles: number of particles in the swarm
        - dim: number of dimensions (24)
        - bounds: (lower_bounds, upper_bounds)
        - inertia: w term
        - cognitive: c1 term
        - social: c2 term
        '''

        self.n_particles = n_particles
        self.dim = dim
        self.bounds = bounds
        
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        self.particles = [Particle(dim, bounds) for _ in range(n_particles)]

        # Initialising Global Best from First Particle

        first_particle = self.particles[0]
        self.global_best_position = first_particle.best_position.copy()
        self.global_best_cost = first_particle.best_cost

        # Checking other particles to update global best

        for p in self.particles[1:]:
            if (p.best_cost < self.global_best_cost):
                self.global_best_cost = p.best_cost
                self.global_best_position = p.best_position.copy()

    def run(self, n_iterations: int, verbose: bool = True):

        '''
        The PSO Main Optimisation Loop

        Parameters:
        - n_iterations: number of PSO Steps
        - verbose: if True, print progress

        Returns:
        - (best_position, best_cost)
        '''
        self.history = []
        self.param_history = []
        self.velocity_history = []

        for it in range(n_iterations):
            
            # Update each particle

            for p in self.particles:

                p.update_velocity(self.global_best_position, self.inertia, self.cognitive, self.social,)
                
                p.velocity_history.append(np.linalg.norm(p.velocity))

                p.update_position()

                p.evaluate()

                if (p.best_cost < self.global_best_cost):
                    self.global_best_cost = p.best_cost
                    self.global_best_position = p.best_position.copy()
                
            self.history.append(self.global_best_cost)
            self.param_history.append(self.global_best_position.copy())

            iter_vel = np.mean([np.linalg.norm(p.velocity) for p in self.particles])
            self.velocity_history.append(iter_vel)
            
            if verbose and (it % 10 == 0 or it == n_iterations - 1):
                print(
                    f"Iteration {it+1}/{n_iterations} | "
                    f"Global Best Cost: {self.global_best_cost: .6f}"
                )
        
        return self.global_best_position, self.global_best_cost




''' import numpy as np

# Import our modules
from cost_function import init_cost_model, cost
from pso import Swarm


# -----------------------
# 1. Create synthetic data
# -----------------------
B_meas = np.array([
    [10., 0., 0.],
    [0., 20., 0.],
    [0., 0., 30.],
    [5., 5., 5.]
])

true_S = np.eye(3)
true_Ks = np.zeros((3, 3))
true_O = np.array([[1.0], [2.0], [-1.0]])
true_Ko = np.zeros((3, 1))

# Build true parameter vector (length 24)
params_true = np.hstack([
    true_S.reshape(-1),
    true_Ks.reshape(-1),
    true_O.flatten(),
    true_Ko.flatten()
])

T = np.array([25., 25., 25., 30.])

# Compute reference output, as if measured
B_ref = ((true_S + true_Ks * T[:, None, None]) @ B_meas[:, :, None]).squeeze() \
        + (true_O.flatten() + true_Ko.flatten() * T[:, None])


# ------------------------------------
# 2. Initialize CalibrationModel inside cost_function
# ------------------------------------
init_cost_model(B_meas, B_ref, T)


# -----------------------
# 3. Prepare PSO parameters
# -----------------------
dim = 24

# Example bounds — wide but reasonable
lower_bounds = -5 * np.ones(dim)
upper_bounds = 5 * np.ones(dim)
bounds = (lower_bounds, upper_bounds)

# Create the swarm
swarm = Swarm(
    n_particles=10,
    dim=dim,
    bounds=bounds,
    inertia=0.7,
    cognitive=1.5,
    social=1.5,
)


# -----------------------
# 4. Run PSO
# -----------------------
best_pos, best_cost = swarm.run(
    n_iterations=30,
    verbose=True
)

print("\n=== PSO Result ===")
print("Best cost:", best_cost)
print("Best parameters:", best_pos)
print("True params:", params_true)
print("Error norm:", np.linalg.norm(best_pos - params_true))

'''