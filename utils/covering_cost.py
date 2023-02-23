import numpy as np


class CoveringCost:
    def __init__(
        self,
        nb_work_days_per_week: int,
        nb_shifts_per_work_day: int,
        nb_nrs_per_shift: int,
    ) -> None:
        self.nb_work_days_per_week = nb_work_days_per_week
        self.nb_shifts_per_work_day = nb_shifts_per_work_day
        self.nb_nrs_per_shift = nb_nrs_per_shift

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
