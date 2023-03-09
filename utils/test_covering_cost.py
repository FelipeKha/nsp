import numpy as np
import pytest

from problem_setup.problem import Problem
from utils.covering_cost import covering_cost


class TestCoveringCost:

    @pytest.fixture
    def problem(self):
        """
        Returns a Problem instance with:
        - Nb nurses: 4
        - Work days: 7
        - Shift per workday: 1
        - Required nurses per shift: 2
        - Max work days per week: 5
        """
        return Problem(
            nb_nurses=4,
            nb_work_days_per_week=7,
            nb_shifts_per_work_day=1,
            target_nb_nrs_per_shift=2,
            nrs_max_work_days_per_week=5,
        )

    # returns type np.int64
    def test_covering_cost_returns_type_np_int64(self, problem):
        assert type(covering_cost(np.zeros((4, 7)), problem)) == np.int64

    # returns 0 for solution
    def test_covering_cost_returns_0_for_zero_solution(self, problem):
        solution = np.ones((2, 7))
        assert covering_cost(solution, problem) == 0

    # returns nb_shifts_per_work_day * nb_work_days_per_week * 2 for zero solution
    def test_covering_cost_returns_required_coverage_for_zero_solution(self, problem):
        solution = np.zeros((4, 7))
        assert covering_cost(solution, problem) == 28

    # throw an error if solution length is not dim x, nb_shifts_per_work_day * nb_work_days_per_week
    def test_covering_cost_throws_error_if_solution_length_is_not_dim_x(self, problem):
        solution = np.zeros((4, 6))
        with pytest.raises(ValueError):
            covering_cost(solution, problem)

    # return positive int if under covered
    def test_covering_cost_returns_positive_int_if_under_covered(self, problem):
        solution = np.ones((2, 7))
        solution[0, 0] = 0
        assert covering_cost(solution, problem) > 0

    # return positive int if over covered
    def test_covering_cost_returns_positive_int_if_over_covered(self, problem):
        solution = np.ones((3, 7))
        assert covering_cost(solution, problem) > 0
