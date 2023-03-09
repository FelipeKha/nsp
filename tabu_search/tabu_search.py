import numpy as np

from problem_setup.problem import Problem

class TabuSearch:
    def __init__(
        self,
        nb_iter: int,
        nb_neighbours: int,
        tabu_limit: int,
        get_random_initial_solution: callable,
        get_neighbour_tabu: callable,
        covering_cost: callable,
    ) -> None:
        self.nb_iter = nb_iter
        self.nb_neighbours = nb_neighbours
        self.tabu_limit = tabu_limit
        self.get_random_initial_solution = get_random_initial_solution
        self.get_neighbour_tabu = get_neighbour_tabu
        self.covering_cost = covering_cost

    def __call__(
            self,
            problem: Problem,
    ) -> tuple[np.ndarray, int, list]:
        initial_solution = self.get_random_initial_solution(problem)
        solution, solution_cost, states = self.tabu_search(
            initial_solution,
            problem,
            )
        self.out = (solution, solution_cost, states)
        return self.out

    def tabu_search(
        self,
        initial_solution: np.ndarray,
        problem: Problem,
    ) -> tuple[np.ndarray, int, list]:
        best_solution = initial_solution
        best_solution_cost = self.covering_cost(best_solution, problem)
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
                self.get_neighbour_tabu(
                    best_solution,
                    problem,
                    self.nb_neighbours,
                    tabu_history,
                    self.tabu_limit,
                    self.covering_cost,
                )
            if best_neighbour_cost <= best_solution_cost:
                best_solution = best_neighbour
                best_solution_cost = best_neighbour_cost
            states.append(best_neighbour_cost)
        return best_solution, best_solution_cost, states