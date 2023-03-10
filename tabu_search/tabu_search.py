import numpy as np

from problem_setup.problem import Problem
from validation import Validation
from utils.iter_next import IterNext

class TabuSearch:
    def __init__(
        self,
        nb_iter: int,
        nb_neighbours: int,
        tabu_limit: int,
        get_random_initial_solution: callable,
        get_neighbour_tabu: callable,
        covering_cost: callable,
        validation: Validation,
        iter_next: IterNext,
    ) -> None:
        self.nb_iter = nb_iter
        self.nb_neighbours = nb_neighbours
        self.tabu_limit = tabu_limit
        self.get_random_initial_solution = get_random_initial_solution
        self.get_neighbour_tabu = get_neighbour_tabu
        self.covering_cost = covering_cost
        self.validation = validation
        self.iter_next = iter_next

    def __call__(
            self,
            problem: Problem,
    ) -> tuple[np.ndarray, int, list]:
        initial_solution = self.get_random_initial_solution(problem)
        out = self.tabu_search(
            initial_solution,
            problem,
            )
        return out

    def tabu_search(
        self,
        initial_solution: np.ndarray,
        problem: Problem,
    ) -> tuple[np.ndarray, int, list]:
        next_iter = True
        iter = 0
        best_solution = initial_solution
        best_solution_cost = self.covering_cost(best_solution, problem)
        states = [best_solution_cost]  # to plot costs through the algo
        tabu_history = {}

        while next_iter:
            iter += 1
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
            
            validate, validation_details = self.validation(
                best_solution,
                best_solution_cost,
                states,
                problem,
            )
            next_iter = self.iter_next.check_if_one_more_iter(
                iter=iter,
                covering_cost=best_solution_cost,
                validate=validate,
                states=states,
            )

        return (
            best_solution, 
            best_solution_cost, 
            states,
            iter,
            validate,
            validation_details,
        )