import numpy as np
import random

from problem_setup.problem import Problem


class SimulatedAnnealing:
    def __init__(
        self,
        nb_iter: int,
        nb_neighbours: int,
        k: int,
        lam: float,
        limit: int,
        get_random_initial_solution: callable,
        get_neighbour: callable,
        covering_cost: callable,
    ) -> None:
        self.nb_iter = nb_iter
        self.nb_neighbours = nb_neighbours
        self.k = k
        self.lam = lam
        self.limit = limit
        self.get_random_initial_solution = get_random_initial_solution
        self.get_neighbour = get_neighbour
        self.covering_cost = covering_cost

    def __call__(
            self,
            problem: Problem,
    ) -> tuple[np.ndarray, int, list]:
        initial_solution = self.get_random_initial_solution(problem)
        best_solution, best_solution_cost, states = \
            self.simulated_annealing(initial_solution, problem)
        return best_solution, best_solution_cost, states
        

    def exp_schedule(
        self,
        t: int,
        k: int = 20,
        lam: float = 0.005,
        limit: int = 100,
    ) -> np.float64:
        out = k * np.exp(-lam * t) if t < limit else 0
        return out

    def simulated_annealing(
        self,
        initial_solution: np.ndarray,
        problem: Problem,
    ) -> tuple[np.ndarray, int, list]:

        current_solution = initial_solution
        current_solution_cost = \
            self.covering_cost(current_solution, problem)
        states = [current_solution_cost]
        for i in range(self.nb_iter):
            print(i)
            T = self.exp_schedule(
                i,
                self.k,
                self.lam,
                self.limit
            )
            candidate_solution, candidate_solution_cost = \
                self.get_neighbour(
                    current_solution,
                    problem, 
                    self.nb_neighbours,
                    self.covering_cost,
                    )
            delta_e = candidate_solution_cost - current_solution_cost
            proba = np.exp(-1 * delta_e / T) if T > 0 else 0
            accept_current_sol = proba > random.uniform(0.0, 1.0)
            if delta_e < 0 or accept_current_sol:
                current_solution = candidate_solution
                current_solution_cost = candidate_solution_cost
            states.append(current_solution_cost)

        return current_solution, current_solution_cost, states