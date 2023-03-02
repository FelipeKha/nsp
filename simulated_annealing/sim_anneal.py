import numpy as np
import random

from utils.covering_cost import CoveringCost
from utils.get_neighbour import GetNeighbour
from utils.get_population import GetPopulation


class SimulatedAnnealing:
    def __init__(
        self,
        nb_nurses: int,
        nb_work_days_per_week: int,
        nb_shifts_per_work_day: int,
        nb_nrs_per_shift: int,
        nrs_max_work_days_per_week: int,
        nb_iter: int,
        nb_neighbours: int,
        k: int,
        lam: float,
        limit: int,
        get_population: GetPopulation,
        get_neighbour: GetNeighbour,
        covering_cost: CoveringCost,
    ) -> None:
        self.nb_nurses = nb_nurses
        self.nb_work_days_per_week = nb_work_days_per_week
        self.nb_shifts_per_work_day = nb_shifts_per_work_day
        self.nb_nrs_per_shift = nb_nrs_per_shift
        self.nrs_max_work_days_per_week = nrs_max_work_days_per_week
        self.nb_iter = nb_iter
        self.nb_neighbours = nb_neighbours
        self.k = k
        self.lam = lam
        self.limit = limit
        self.get_population = get_population
        self.get_neighbour = get_neighbour
        self.covering_cost = covering_cost

    def exp_schedule(
        self,
        t,
        k: int = 20,
        lam: float = 0.005,
        limit: int = 100,
    ) -> float:
        out = k * np.exp(-lam * t) if t < limit else 0
        return out

    def simulated_annealing(
        self,
        initial_solution: np.ndarray,
    ) -> tuple[np.ndarray, int, list]:

        current_solution = initial_solution
        current_solution_cost = \
            self.covering_cost.covering_cost(current_solution)
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
                self.get_neighbour.get_neighbour(
                    current_solution, self.nb_neighbours)
            delta_e = candidate_solution_cost - current_solution_cost
            proba = np.exp(-1 * delta_e / T) if T > 0 else 0
            accept_current_sol = proba > random.uniform(0.0, 1.0)
            if delta_e < 0 or accept_current_sol:
                current_solution = candidate_solution
                current_solution_cost = candidate_solution_cost
            states.append(current_solution_cost)

        return current_solution, current_solution_cost, states

    def search_solution(self):
        initial_solution = self.get_population.get_random_initial_solution()
        best_solution, best_solution_cost, states = \
            self.simulated_annealing(initial_solution)
        return best_solution, best_solution_cost, states
