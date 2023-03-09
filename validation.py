import numpy as np

from problem_setup.problem import Problem


class Validation:
    def __init__(
        self,
    ) -> None:
        self.validation_object = {}
        self.overall_validation = False

    def __call__(
        self,
        solution: np.ndarray,
        solution_cost: int,
        states: list,
        problem: Problem,
    ) -> tuple[bool, dict]:
        self.validation_object['solution_shape_is_correct'] = \
            self.validate_solution_shape(solution, problem)
        self.validation_object['solution_complies_with_max_work_days'] = \
            self.validate_solution_comply_with_max_work_days(solution, problem)
        self.validation_object['solution_complies_min_required_nurses_per_shift'] = \
            self.validate_solution_comply_with_min_required_nurses_per_shift(
                solution, 
                problem,
                )
        self.validation_object['solution_cost_is_zero'] = \
            self.validate_solution_cost(solution_cost)
        self.validation_object['solution_cost_is_min_cost'] = \
            self.validate_solution_cost_is_min_cost(solution_cost, states)
        self.overall_validation = self.validate_overall()
        return self.overall_validation, self.validation_object

    def validate_solution_shape(
            self,
            solution: np.ndarray,
            problem: Problem,
    ) -> bool:
        return solution.shape == (
            problem.nb_nurses,
            problem.nb_work_days_per_week * problem.nb_shifts_per_work_day
        )

    def validate_solution_comply_with_max_work_days(
            self,
            solution: np.ndarray,
            problem: Problem,
    ) -> bool:
        return solution.sum(axis=1).max() <= problem.nrs_max_work_days_per_week

    def validate_solution_comply_with_min_required_nurses_per_shift(
            self,
            solution: np.ndarray,
            problem: Problem,
    ) -> bool:
        return solution.sum(axis=0).min() >= problem.target_nb_nrs_per_shift

    def validate_solution_cost(self, solution_cost: np.float64) -> bool:
        return solution_cost == 0

    def validate_solution_cost_is_min_cost(
            self, 
            solution_cost: np.float64, 
            states: list,
            ) -> bool:
        return solution_cost == min(states)

    def validate_overall(self) -> bool:
        return all(self.validation_object.values())
