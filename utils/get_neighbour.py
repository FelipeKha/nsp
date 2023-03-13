import numpy as np

from problem_setup.problem import Problem
# from utils.covering_cost import CoveringCost
from utils.covering_cost import covering_cost

#######################
# Class version

# class GetNeighbour:
#     def __init__(
#         self,
#         nb_nurses: int, 
#         nb_work_days_per_week: int,
#         nb_shifts_per_work_day: int, 
#         nrs_max_work_days_per_week: int,
#         covering_cost: CoveringCost,
#     ) -> None:
#         self.nb_nurses = nb_nurses
#         self.nb_work_days_per_week = nb_work_days_per_week
#         self.nb_shifts_per_work_day = nb_shifts_per_work_day
#         self.nrs_max_work_days_per_week = nrs_max_work_days_per_week
#         self.covering_cost = covering_cost
    
#     def get_neighbour(
#         self,
#         solution: np.ndarray,
#         nb_neighbours: int,
#     ) -> tuple[np.ndarray, np.int64]:
#         """
#         Given a solution of dim (nb_nurses, nb_shifts), generate nb_neighbours 
#         neighbours, and return the best one.
#         A neighbour is a solution where one nurse has changed one shift.
#         """
#         best_neighbour = np.empty(
#             (
#                 self.nb_nurses,
#                 self.nb_work_days_per_week * self.nb_shifts_per_work_day
#             ),
#             dtype=int
#         )
#         best_neighbour_cost = np.inf

#         for _ in range(nb_neighbours):
#             neighbour = solution.copy()
#             nrs = np.random.randint(self.nb_nurses)
#             shift = np.random.randint(
#                 self.nb_work_days_per_week * self.nb_shifts_per_work_day
#             )
#             neighbour[nrs, shift] = 1 - neighbour[nrs, shift]
#             if neighbour[nrs].sum() > self.nrs_max_work_days_per_week:
#                 continue
#             neighbour_cost = self.covering_cost.covering_cost(neighbour)
#             if neighbour_cost < best_neighbour_cost:
#                 best_neighbour = neighbour
#                 best_neighbour_cost = neighbour_cost

#         return best_neighbour, best_neighbour_cost

#     def get_neighbour_tabu(
#         self,
#         solution: np.ndarray,
#         nb_neighbours: int,
#         tabu_history: dict,
#         tabu_limit: int,
#     ) -> tuple[np.ndarray, np.int64, dict]:
#         """
#         Given a solution of dim (nb_nurses, nb_shifts), generate nb_neighbours 
#         neighbours, check them aginst the tabu list, and return the best one and 
#         the updated tabu list.
#         A neighbour is a solution where one nurse has changed one shift.
#         """
#         best_neighbour = np.empty(
#             (
#                 self.nb_nurses,
#                 self.nb_work_days_per_week * self.nb_shifts_per_work_day
#             ),
#             dtype=int
#         )
#         best_neighbour_cost = np.inf

#         for _ in range(nb_neighbours):
#             neighbour = solution.copy()
#             nrs = np.random.randint(self.nb_nurses)
#             shift = np.random.randint(
#                 self.nb_work_days_per_week * self.nb_shifts_per_work_day
#             )
#             neighbour[nrs, shift] = 1 - neighbour[nrs, shift]
#             if (tuple(map(tuple, neighbour)) in tabu_history) or \
#                 neighbour[nrs].sum() > self.nrs_max_work_days_per_week:
#                 continue
#             neighbour_cost = self.covering_cost.covering_cost(neighbour)
#             if neighbour_cost < best_neighbour_cost:
#                 best_neighbour = neighbour
#                 best_neighbour_cost = neighbour_cost
#                 tabu_history[tuple(map(tuple, best_neighbour))] = tabu_limit
            
#         return best_neighbour, best_neighbour_cost, tabu_history

######################
# Function version

def get_neighbour(
    solution: np.ndarray,
    problem: Problem,
    nb_neighbours: int,
    covering_cost: callable,
) -> tuple[np.ndarray, np.int64]:
    """
    Given a solution of dim (nb_nurses, nb_shifts), generate nb_neighbours 
    neighbours, and return the best one.
    A neighbour is a solution where one nurse has changed one shift.
    """
    best_neighbour = np.empty(
        (
            problem.nb_nurses,
            problem.nb_work_days_per_week * problem.nb_shifts_per_work_day
        ),
        dtype=int
    )
    best_neighbour_cost = np.inf

    for _ in range(nb_neighbours):
        neighbour = solution.copy()
        nrs = np.random.randint(problem.nb_nurses)
        shift = np.random.randint(
            problem.nb_work_days_per_week * problem.nb_shifts_per_work_day
        )
        neighbour[nrs, shift] = 1 - neighbour[nrs, shift]
        if neighbour[nrs].sum() > problem.nrs_max_work_days_per_week:
            continue
        neighbour_cost = covering_cost(neighbour, problem)
        if neighbour_cost < best_neighbour_cost:
            best_neighbour = neighbour
            best_neighbour_cost = neighbour_cost

    return best_neighbour, best_neighbour_cost

def get_neighbour_tabu(
    solution: np.ndarray,
    problem: Problem,
    nb_neighbours: int,
    tabu_history: dict,
    tabu_limit: int,
    covering_cost: callable,
) -> tuple[np.ndarray, np.int64, dict]:
    """
    Given a solution of dim (nb_nurses, nb_shifts), generate nb_neighbours 
    neighbours, check them aginst the tabu list, and return the best one and 
    the updated tabu list.
    A neighbour is a solution where one nurse has changed one shift.
    """
    best_neighbour = np.empty(
        (
            problem.nb_nurses,
            problem.nb_work_days_per_week * problem.nb_shifts_per_work_day
        ),
        dtype=int
    )
    best_neighbour_cost = np.inf

    for _ in range(nb_neighbours):
        neighbour = solution.copy()
        nrs = np.random.randint(problem.nb_nurses)
        shift = np.random.randint(
            problem.nb_work_days_per_week * problem.nb_shifts_per_work_day
        )
        neighbour[nrs, shift] = 1 - neighbour[nrs, shift]
        if (tuple(map(tuple, neighbour)) in tabu_history) or \
            neighbour[nrs].sum() > problem.nrs_max_work_days_per_week:
            continue
        neighbour_cost = covering_cost(neighbour, problem)
        if neighbour_cost < best_neighbour_cost:
            best_neighbour = neighbour
            best_neighbour_cost = neighbour_cost
            tabu_history[tuple(map(tuple, best_neighbour))] = tabu_limit
        
    return best_neighbour, best_neighbour_cost, tabu_history

        #--------------
        # best_neighbour = np.empty(solution.shape, dtype=int)
        # best_neighbour_cost = np.inf

        # for nrs in range(solution.shape[0]):
        #     for shift in range(solution.shape[1]):
        #         neighbour = solution.copy()
        #         neighbour[nrs, shift] = 1 - neighbour[nrs, shift]
        #         if (tuple(map(tuple, neighbour)) in tabu_history) or \
        #                 (neighbour[nrs].sum() > self.nrs_max_work_days_per_week):
        #             continue
        #         neighbour_cost = self.covering_cost.covering_cost(neighbour)
        #         if neighbour_cost < best_neighbour_cost:
        #             best_neighbour = neighbour
        #             best_neighbour_cost = neighbour_cost
        #             tabu_history[tuple(
        #                 map(tuple, best_neighbour))] = self.tabu_limit
        # return best_neighbour, best_neighbour_cost, tabu_history
