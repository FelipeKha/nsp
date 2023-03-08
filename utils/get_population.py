import numpy as np


class GetPopulation:
    def __init__(
        self, 
        nb_nurses: int, 
        nb_work_days_per_week: int,
        nb_shifts_per_work_day: int, 
        nrs_max_work_days_per_week: int,
    ) -> None:
        self.nb_nurses = nb_nurses
        self.nb_work_days_per_week = nb_work_days_per_week
        self.nb_shifts_per_work_day = nb_shifts_per_work_day
        self.nrs_max_work_days_per_week = nrs_max_work_days_per_week


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

    def get_initial_population(self, pop_size) -> np.ndarray:
        """
        Given the population size, the number of nurses, the number of shifts, 
        and the maximum number of work days per week, returns an initial 
        population of solutions.
        Output is a numpy array of shape (pop_size, nb_nurses, nb_shifts) where 
        each element is either 0 or 1.
        """
        out = np.empty(
            (
                pop_size,
                self.nb_nurses,
                self.nb_work_days_per_week * self.nb_shifts_per_work_day
            ),
            dtype=int
        )

        for pop in range(pop_size):
            out[pop] = self.get_random_initial_solution()
        return out


####################
# Function version

# def get_random_nurse_schedule(
#         nb_work_days_per_week: int,
#         nb_shifts_per_work_day: int,
#         nrs_max_work_days_per_week: int,
# ) -> np.ndarray:
#     nurses_worked_days_per_week = np.random.randint(
#         0,
#         nrs_max_work_days_per_week + 1,
#         1,
#     )
#     nurse_schedule = np.concatenate([
#         np.ones(nurses_worked_days_per_week, dtype=int),
#         np.zeros(
#             nb_work_days_per_week * nb_shifts_per_work_day -
#             nurses_worked_days_per_week,
#             dtype=int
#         ),
#     ])
#     nurse_schedule = np.random.permutation(nurse_schedule)
#     return nurse_schedule

# def get_random_initial_solution(
#         nb_nurses: int,
#         nb_work_days_per_week: int,
#         nb_shifts_per_work_day: int,
#         nrs_max_work_days_per_week: int,
# ) -> np.ndarray:
#     """
#     Given the number of nurses, the number of shifts, and the maximum number of 
#     work days per week, returns a random initial solution.
#     Output is a numpy array of shape (nb_nurses, nb_shifts) where each element 
#     is either 0 or 1.
#     """
#     out = np.empty(
#         (
#             nb_nurses,
#             nb_work_days_per_week * nb_shifts_per_work_day
#         ),
#         dtype=int
#     )

#     for nrs in range(nb_nurses):
#         nurse_schedule = get_random_nurse_schedule(
#             nb_work_days_per_week,
#             nb_shifts_per_work_day,
#             nrs_max_work_days_per_week,
#         )
#         out[nrs] = nurse_schedule
#     return out

# def get_initial_population(
#         pop_size: int,
#         nb_nurses: int,
#         nb_work_days_per_week: int,
#         nb_shifts_per_work_day: int,
#         nrs_max_work_days_per_week: int,
# ) -> np.ndarray:
#     """
#     Given the population size, the number of nurses, the number of shifts, 
#     and the maximum number of work days per week, returns an initial 
#     population of solutions.
#     Output is a numpy array of shape (pop_size, nb_nurses, nb_shifts) where 
#     each element is either 0 or 1.
#     """
#     out = np.empty(
#         (
#             pop_size,
#             nb_nurses,
#             nb_work_days_per_week * nb_shifts_per_work_day
#         ),
#         dtype=int
#     )

#     for pop in range(pop_size):
#         out[pop] = get_random_initial_solution(
#             nb_nurses,
#             nb_work_days_per_week,
#             nb_shifts_per_work_day,
#             nrs_max_work_days_per_week,
#         )
#     return out