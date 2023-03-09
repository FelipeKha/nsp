import numpy as np

from utils.covering_cost import CoveringCost
from utils.get_neighbour import GetNeighbour
from utils.get_population import GetPopulation


class TabuSearch:
    def __init__(
        self,
        nb_iter: int,
        nb_neighbours: int,
        tabu_limit: int,
        GetPopulation: GetPopulation,
        GetNeighbour: GetNeighbour,
        CoveringCost: CoveringCost,
    ) -> None:
        self.nb_iter = nb_iter
        self.nb_neighbours = nb_neighbours
        self.tabu_limit = tabu_limit
        self.GetPopulation = GetPopulation
        self.GetNeighbour = GetNeighbour
        self.CoveringCost = CoveringCost

    def __call__(
            self,
            nb_nurses: int,
            nb_work_days_per_week: int,
            nb_shifts_per_work_day: int,
            nb_nrs_per_shift: int,
            nrs_max_work_days_per_week: int,
    ) -> tuple[np.ndarray, int, list]:
        covering_cost = self.CoveringCost(
            nb_work_days_per_week,
            nb_shifts_per_work_day,
            nb_nrs_per_shift,
        )
        get_neighbour = self.GetNeighbour(
            nb_nurses,
            nb_work_days_per_week,
            nb_shifts_per_work_day,
            nrs_max_work_days_per_week,
            covering_cost,
        )
        get_population = self.GetPopulation(
            nb_nurses,
            nb_work_days_per_week,
            nb_shifts_per_work_day,
            nrs_max_work_days_per_week,
        )

        initial_solution = get_population.get_random_initial_solution()
        solution, solution_cost, states = self.tabu_search(
            initial_solution,
            covering_cost,
            get_neighbour,
        )
        self.out = (solution, solution_cost, states)
        return self.out

    def tabu_search(
        self,
        initial_solution: np.ndarray,
        covering_cost: CoveringCost,
        get_neighbour: GetNeighbour,
    ) -> tuple[np.ndarray, int, list]:
        best_solution = initial_solution
        best_solution_cost = covering_cost.covering_cost(best_solution)
        states = [best_solution_cost]  # to plot costs through the algo
        tabu_history = {}

        for iter in range(self.nb_iter):
            print('iter', iter)
            for sol in tabu_history:
                tabu_history[sol] -= 1
            tabu_history = {
                sol: tabu_history[sol] for sol in tabu_history if tabu_history[sol] > 0
            }
            best_neighbour, best_neighbour_cost, tabu_history = \
                get_neighbour.get_neighbour_tabu(
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