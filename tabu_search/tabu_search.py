import numpy as np

from utils.covering_cost import CoveringCost
from utils.get_neighbour import GetNeighbour
from utils.get_population import GetPopulation


class TabuSearch:
    def __init__(
        self,
        nb_nurses: int,
        nb_work_days_per_week: int,
        nb_shifts_per_work_day: int,
        nb_nrs_per_shift: int,
        nrs_max_work_days_per_week: int,
        nb_iter: int,
        nb_neighbours: int,
        tabu_limit: int,
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
        self.tabu_limit = tabu_limit
        self.get_population = get_population
        self.get_neighbour = get_neighbour
        self.covering_cost = covering_cost

    def tabu_search(
        self,
        initial_solution: np.ndarray,
    ) -> tuple[np.ndarray, int, list]:
        best_solution = initial_solution
        best_solution_cost = self.covering_cost.covering_cost(best_solution)
        states = [best_solution_cost]  # to plot costs through the algo
        tabu_history = {}

        for iter in range(self.nb_iter):
            print('iter', iter)
            # reduce counter for all tabu
            for sol in tabu_history:
                tabu_history[sol] -= 1
            tabu_history = {
                sol: tabu_history[sol] for sol in tabu_history if tabu_history[sol] > 0
            }
            best_neighbour, best_neighbour_cost, tabu_history = \
                self.get_neighbour.get_neighbour_tabu(
                    best_solution,
                    self.nb_neighbours,
                    tabu_history,
                    self.tabu_limit,
                )
            if best_neighbour_cost <= best_solution_cost:
                best_solution = best_neighbour
                best_solution_cost = best_neighbour_cost
            states.append(best_neighbour_cost)
        return best_solution, best_solution_cost, states

    def search_solution(self) -> np.ndarray:
        # get initial random solution
        initial_solution = self.get_population.get_random_initial_solution()

        # tabu search
        # tabu list, in type dictionary (solution: nb of remaining iter in tabu list)
        solution, solution_cost, states = self.tabu_search(initial_solution)
        return solution, solution_cost, states
