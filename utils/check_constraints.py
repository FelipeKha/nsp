import numpy as np

from problem_setup.problem import Problem

###################
# Class version

# class CheckConstraints:
#     def __init__(
#         self,
#     ) -> None:
#         pass

#     def check_solution_for_max_work_days_per_week(
#         self,
#         solution: np.ndarray,
#         target_nb_nurse_per_shift: int,
#         max_work_days_per_week: int,
#     ) -> np.ndarray:
#         """
#         Given a solution of dim (nb_nurses, nb_shifts), check that the sum of
#         number of shifts per week is less or equal to the max number of shifts 
#         for each nurse. If not, the nurse which does not comply with the max
#         number of shifts has shifts removed:
#         - If some shifts are overcovered, it will start by removing shifts here
#         - Else, it will remove random shifts until the max number of shifts is
#         complied with.
#         Returns updated solution of dim (nb_nurses, nb_shifts).
#         """
#         nb_shifts_per_nurse = np.sum(solution, axis=1)
#         def func(x): return max(0, x - max_work_days_per_week)
#         func = np.vectorize(func)
#         nb_shifts_to_remove = func(nb_shifts_per_nurse)
#         while nb_shifts_to_remove.sum() > 0:
#             nurses_to_adjust = np.stack(
#                 np.where(nb_shifts_to_remove > 0),
#                 axis=0
#             )[0]
#             n = nurses_to_adjust[0]
#             shifts_cover = np.sum(solution, axis=0)
#             shifts_overcover = np.where(
#                 shifts_cover > target_nb_nurse_per_shift
#                 )[0]
#             if shifts_overcover.size > 0:
#                 for s in shifts_overcover:
#                     if solution[n, s] == 1:
#                         solution[n, s] = 0
#                         nb_shifts_to_remove[n] -= 1
#                         if nb_shifts_to_remove[n] == 0:
#                             break
#                     else:
#                         shifts_to_remove = np.random.choice(
#                             np.where(solution[n] == 1)[0],
#                             nb_shifts_to_remove[n],
#                             replace=False,
#                         )
#                         for s in shifts_to_remove:
#                             solution[n, s] = 0
#                             nb_shifts_to_remove[n] -= 1
#                     if nb_shifts_to_remove[n] == 0:
#                         break
#             else:
#                 shifts_to_remove = np.random.choice(
#                     np.where(solution[n] == 1)[0],
#                     nb_shifts_to_remove[n],
#                     replace=False,
#                 )
#                 for s in shifts_to_remove:
#                     solution[n, s] = 0
#                     nb_shifts_to_remove[n] -= 1
#         return solution

#     def check_population_for_max_days_per_week(
#             self,
#             population: np.ndarray,
#             target_nb_nurse_per_shift: int,
#             max_work_days_per_week: int,
#     ) -> np.ndarray:
#         """
#         Given population of dim (pop_size, nb_nurses, nb_shifts), apply 
#         check_solution_for_max_work_days_per_week() to each solution that does 
#         not comply with max number of shift per week constraint.
#         Returns updated population. 
#         """
#         nb_shift_per_sol = np.sum(population, axis=2)
#         def func(x): return max(0, x - max_work_days_per_week)
#         func = np.vectorize(func)
#         nb_shift_to_remove = func(nb_shift_per_sol)
#         nb_shift_to_remove_per_sol = nb_shift_to_remove.sum(axis=1)
#         sol_to_adjust = np.stack(
#                         np.where(nb_shift_to_remove_per_sol > 0), axis=0)[0]
#         for sol_i in sol_to_adjust:
#             population[sol_i] = self.check_solution_for_max_work_days_per_week(
#                 population[sol_i],
#                 target_nb_nurse_per_shift,
#                 max_work_days_per_week,
#             )
#         return population
    

###################
# Function version

def check_solution_for_max_work_days_per_week(
    solution: np.ndarray,
    problem: Problem,
) -> np.ndarray:
    """
    Given a solution of dim (nb_nurses, nb_shifts), check that the sum of
    number of shifts per week is less or equal to the max number of shifts 
    for each nurse. If not, the nurse which does not comply with the max
    number of shifts has shifts removed:
    - If some shifts are overcovered, it will start by removing shifts here
    - Else, it will remove random shifts until the max number of shifts is
    complied with.
    Returns updated solution of dim (nb_nurses, nb_shifts).
    """
    nb_shifts_per_nurse = np.sum(solution, axis=1)
    def func(x): return max(0, x - problem.nrs_max_work_days_per_week)
    func = np.vectorize(func)
    nb_shifts_to_remove = func(nb_shifts_per_nurse)
    while nb_shifts_to_remove.sum() > 0:
        nurses_to_adjust = np.stack(
            np.where(nb_shifts_to_remove > 0),
            axis=0
        )[0]
        n = nurses_to_adjust[0]
        shifts_cover = np.sum(solution, axis=0)
        shifts_overcover = np.where(
            shifts_cover > problem.target_nb_nrs_per_shift
            )[0]
        if shifts_overcover.size > 0:
            for s in shifts_overcover:
                if solution[n, s] == 1:
                    solution[n, s] = 0
                    nb_shifts_to_remove[n] -= 1
                    if nb_shifts_to_remove[n] == 0:
                        break
                else:
                    shifts_to_remove = np.random.choice(
                        np.where(solution[n] == 1)[0],
                        nb_shifts_to_remove[n],
                        replace=False,
                    )
                    for s in shifts_to_remove:
                        solution[n, s] = 0
                        nb_shifts_to_remove[n] -= 1
                if nb_shifts_to_remove[n] == 0:
                    break
        else:
            shifts_to_remove = np.random.choice(
                np.where(solution[n] == 1)[0],
                nb_shifts_to_remove[n],
                replace=False,
            )
            for s in shifts_to_remove:
                solution[n, s] = 0
                nb_shifts_to_remove[n] -= 1
    return solution

def check_population_for_max_days_per_week(
        population: np.ndarray,
        problem: Problem,
) -> np.ndarray:
    """
    Given population of dim (pop_size, nb_nurses, nb_shifts), apply 
    check_solution_for_max_work_days_per_week() to each solution that does 
    not comply with max number of shift per week constraint.
    Returns updated population. 
    """
    nb_shift_per_sol = np.sum(population, axis=2)
    def func(x): return max(0, x - problem.nrs_max_work_days_per_week)
    func = np.vectorize(func)
    nb_shift_to_remove = func(nb_shift_per_sol)
    nb_shift_to_remove_per_sol = nb_shift_to_remove.sum(axis=1)
    sol_to_adjust = np.stack(
                    np.where(nb_shift_to_remove_per_sol > 0), axis=0)[0]
    for sol_i in sol_to_adjust:
        population[sol_i] = check_solution_for_max_work_days_per_week(
            population[sol_i],
            problem,
        )
    return population