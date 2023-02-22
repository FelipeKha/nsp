import numpy as np


class TabuSearch:
    def __init__(
        self,
        nb_nurses: int,
        nb_work_days_per_week: int,
        nb_shifts_per_work_day: int,
        nb_nrs_per_shift: int,
        nrs_max_work_days_per_week: int,
        nb_iter: int,
        tabu_limit: int,
    ):
        self.nb_nurses = nb_nurses
        self.nb_work_days_per_week = nb_work_days_per_week
        self.nb_shifts_per_work_day = nb_shifts_per_work_day
        self.nb_nrs_per_shift = nb_nrs_per_shift
        self.nrs_max_work_days_per_week = nrs_max_work_days_per_week
        self.nb_iter = nb_iter
        self.tabu_limit = tabu_limit

    def get_random_nurse_schedule(self) -> np.ndarray:
        nurses_worked_days_per_week = np.random.randint(
            0,
            self.nrs_max_work_days_per_week + 1,
            1,
        )
        nurse_schedule = np.concatenate([
            np.ones(nurses_worked_days_per_week, dtype=int),
            np.zeros(
                self.nb_work_days_per_week * self.nb_shifts_per_work_day -
                nurses_worked_days_per_week,
                dtype=int
            ),
        ])
        nurse_schedule = np.random.permutation(nurse_schedule)
        return nurse_schedule

    def get_random_initial_solution(self) -> np.ndarray:
        """
        Given the number of nurses, the number of shifts, and the maximum number of 
        work days per week, returns a random initial solution.
        Output is a numpy array of shape (nb_nurses, nb_shifts) where each element 
        is either 0 or 1.
        """
        out = np.empty(
            (
                self.nb_nurses,
                self.nb_work_days_per_week * self.nb_shifts_per_work_day
            ),
            dtype=int
        )

        for nrs in range(self.nb_nurses):
            nurse_schedule = self.get_random_nurse_schedule()
            out[nrs] = nurse_schedule
        return out

    def covering_cost(self, solution: np.ndarray) -> int:
        """
        Given a solution, returns the covering cost.
        """
        required_coverage_per_shift = np.full(
            (self.nb_work_days_per_week * self.nb_shifts_per_work_day),
            self.nb_nrs_per_shift,
            dtype=int,
        )
        coverage_per_shift = solution.sum(axis=0)
        # coverage_cost = np.sum(np.maximum(0, required_coverage_per_shift - coverage_per_shift))
        coverage_cost = np.sum(
            np.square(required_coverage_per_shift - coverage_per_shift), axis=0)
        return coverage_cost

    def get_best_neighbour(
        self, 
        solution: np.ndarray, 
        tabu_history: dict,
    ) -> tuple[np.ndarray, dict]:
        # generate a list of all neighbours
        # a neighbour is a solution where one nurse has changed one shift
        best_neighbour = np.empty(solution.shape, dtype=int)
        best_neighbour_cost = np.inf

        for nrs in range(solution.shape[0]):
            for shift in range(solution.shape[1]):
                neighbour = solution.copy()
                neighbour[nrs, shift] = 1 - neighbour[nrs, shift]
                if (tuple(map(tuple, neighbour)) in tabu_history) or \
                    (neighbour[nrs].sum() > self.nrs_max_work_days_per_week):
                    continue
                neighbour_cost = self.covering_cost(neighbour)
                if neighbour_cost < best_neighbour_cost:
                    best_neighbour = neighbour
                    best_neighbour_cost = neighbour_cost
                    tabu_history[tuple(map(tuple, best_neighbour))] = self.tabu_limit
        return best_neighbour, best_neighbour_cost, tabu_history

    def tabu_search(
        self,
        initial_solution: np.ndarray,
    ) -> tuple[np.ndarray, int, list]:
        best_solution = initial_solution
        best_solution_cost = self.covering_cost(best_solution)
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
            best_neighbour, best_neighbour_cost, tabu_history = self.get_best_neighbour(
                best_solution,
                tabu_history,
            )
            if best_neighbour_cost <= best_solution_cost:
                best_solution = best_neighbour
                best_solution_cost = best_neighbour_cost
            states.append(best_neighbour_cost)
        return best_solution, best_solution_cost, states

    def search_solution(self) -> np.ndarray:
        # get initial random solution
        initial_solution = self.get_random_initial_solution()

        # tabu search
        # tabu list, in type dictionary (solution: nb of remaining iter in tabu list)
        tabu_history = {}
        solution, solution_cost, states = self.tabu_search(initial_solution)
        return solution, solution_cost, states