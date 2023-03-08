import numpy as np
import pytest

from utils.covering_cost import CoveringCost

class TestCoveringCost:


    @pytest.fixture
    def covering_cost(self):
        """
        Returns a CoveringCost instance with:
        - Work days: 7
        - Shift per workday: 1
        - Required nurses per shift: 2
        """
        return CoveringCost(
            nb_work_days_per_week=7,
            nb_shifts_per_work_day=1,
            nb_nrs_per_shift=2,
        )
    
    # returns type np.int64
    def test_covering_cost_returns_type_np_int64(self, covering_cost):
        assert type(covering_cost.covering_cost(np.zeros((4,7)))) == np.int64
    
    # returns 0 for solution
    def test_covering_cost_returns_0_for_zero_solution(self, covering_cost):
        solution = np.ones((2,7))
        assert covering_cost.covering_cost(solution) == 0
    
    # returns nb_shifts_per_work_day * nb_work_days_per_week * 2 for zero solution
    def test_covering_cost_returns_required_coverage_for_zero_solution(self, covering_cost):
        solution = np.zeros((4,7))
        assert covering_cost.covering_cost(solution) == 28

    # throw an error if solution length is not dim x, nb_shifts_per_work_day * nb_work_days_per_week
    def test_covering_cost_throws_error_if_solution_length_is_not_dim_x(self, covering_cost):
        solution = np.zeros((4,6))
        with pytest.raises(ValueError):
            covering_cost.covering_cost(solution)
    
    # return positive int if under covered
    def test_covering_cost_returns_positive_int_if_under_covered(self, covering_cost):
        solution = np.ones((2, 7))
        solution[0,0] = 0
        assert covering_cost.covering_cost(solution) > 0

    # return positive int if over covered
    def test_covering_cost_returns_positive_int_if_over_covered(self, covering_cost):
        solution = np.ones((3, 7))
        assert covering_cost.covering_cost(solution) > 0