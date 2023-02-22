import numpy as np


class Validation:
    def __init__(
        self,
        nb_nurses: int,
        nb_work_days_per_week: int,
        nb_shifts_per_work_day: int,
        nb_nrs_per_shift: int,
        nrs_max_work_days_per_week: int,
    ):
        self.nb_nurses = nb_nurses
        self.nb_work_days_per_week = nb_work_days_per_week
        self.nb_shifts_per_work_day = nb_shifts_per_work_day
        self.nb_nrs_per_shift = nb_nrs_per_shift
        self.nrs_max_work_days_per_week = nrs_max_work_days_per_week
        self.validation_object = {}
        self.overall_validation = False

    def validate_solution(self, solution, solution_cost, states):
        self.validation_object['solution_shape_is_correct'] = \
            self.validate_solution_shape(solution)
        self.validation_object['solution_complies_with_max_work_days'] = \
            self.validate_solution_comply_with_max_work_days(solution)
        self.validation_object['solution_complies_min_required_nurses_per_shift'] = \
            self.validate_solution_comply_with_min_required_nurses_per_shift(solution)
        self.validation_object['solution_cost_is_zero'] = \
            self.validate_solution_cost(solution_cost)
        self.validation_object['solution_cost_is_min_cost'] = \
            self.validate_solution_cost_is_min_cost(solution_cost, states)
        self.overall_validation = self.validate_overall()
        return self.validation_object, self.overall_validation

    def validate_solution_shape(self, solution):
        return solution.shape == (
            self.nb_nurses,
            self.nb_work_days_per_week * self.nb_shifts_per_work_day
        )

    def validate_solution_comply_with_max_work_days(self, solution):
        return solution.sum(axis=1).max() <= self.nrs_max_work_days_per_week

    def validate_solution_comply_with_min_required_nurses_per_shift(self, solution):
        return solution.sum(axis=0).min() >= self.nb_nrs_per_shift

    def validate_solution_cost(self, solution_cost):
        return solution_cost == 0

    def validate_solution_cost_is_min_cost(self, solution_cost, states):
        return solution_cost == min(states)

    def validate_overall(self):
        return all(self.validation_object.values())