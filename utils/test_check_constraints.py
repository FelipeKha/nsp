import numpy as np
import pytest

from utils.check_constraints import CheckConstraints


class TestCheckConstraints:

    @pytest.fixture
    def check_constraints(self):
        return CheckConstraints()

    def is_solution(self, solution):
        """
        Returns True if solution is a solution. Being a solution means:
        - solution is an ndarray
        - solution has the expected shape (nb_nurses, nb_work_days_per_week * 
        nb_shifts_per_work_day)
        - solution is binary
        """
        if type(solution) != np.ndarray:
            return False
        if solution.shape != (4, 7):
            return False
        if not np.array_equal(solution, solution.astype(bool)):
            return False
        return True

    # check_solution_for_max_work_days_per_week
    # returns a ndarray
    def test_check_solution_for_max_work_days_per_week_returns_ndarray(self, check_constraints):
        solution = np.random.randint(0, 2, size=(4, 7))
        out = check_constraints.check_solution_for_max_work_days_per_week(
            solution, 2, 5)
        assert type(out) == np.ndarray

    # returns a ndarray or dim (nb_nurses, nb_shifts)
    def test_check_solution_for_max_work_days_per_week_returns_ndarray_of_dim_nb_nurses_nb_shifts(self, check_constraints):
        solution = np.random.randint(0, 2, size=(4, 7))
        out = check_constraints.check_solution_for_max_work_days_per_week(
            solution, 2, 5)
        assert out.shape == (4, 7)

    # each item of output is either 0 or 1
    def test_check_solution_for_max_work_days_per_week_each_item_of_output_is_0_or_1(self, check_constraints):
        solution = np.random.randint(0, 2, size=(4, 7))
        out = check_constraints.check_solution_for_max_work_days_per_week(
            solution, 2, 5)
        assert np.array_equal(out, out.astype(bool))

    # each item of output is a solution
    def test_check_solution_for_max_work_days_per_week_each_item_of_output_is_a_solution(self, check_constraints):
        solution = np.random.randint(0, 2, size=(4, 7))
        out = check_constraints.check_solution_for_max_work_days_per_week(
            solution, 2, 5)
        assert self.is_solution(out)

    # each item of output comply with the max work days per week constraint
    def test_check_solution_for_max_work_days_per_week_each_item_of_output_comply_with_the_max_work_days_per_week_constraint(
            self,
            check_constraints,
    ):
        solution = np.random.randint(0, 2, size=(4, 7))
        out = check_constraints.check_solution_for_max_work_days_per_week(
            solution, 2, 5)
        assert out.sum(axis=1).max() <= 5

    # if input has a solution not comlying with the max work days per week,
    # removes shifts from overworked nurse
    def test_check_solution_for_max_work_days_per_week_if_input_has_a_solution_not_complying_with_the_max_work_days_per_week_removes_shifts_from_overworked_nurse(
            self,
            check_constraints,
    ):
        solution = np.zeros((4, 7), dtype=int)
        for i in range(6):
            solution[0, i] = 1
        out = check_constraints.check_solution_for_max_work_days_per_week(
            solution, 2, 5)
        assert out.sum(axis=1).max() <= 5

    # if input has a solution not comlying with the max work days per week, and
    # overworked nurse shift can be remove from overcovred shift, it will remove this in priority
    def test_check_solution_for_max_work_days_per_week_if_input_has_a_solution_not_complying_with_the_max_work_days_per_week_and_overworked_nurse_shift_can_be_remove_from_overcovred_shift_it_will_remove_this_in_priority(
            self,
            check_constraints,
    ):
        solution = np.zeros((4, 7), dtype=int)
        for i in range(6):
            solution[0, i] = 1
        for i in range(1, 4):
            solution[i, 0] = 1
        out = check_constraints.check_solution_for_max_work_days_per_week(
            solution, 2, 5)
        target_solution = solution.copy()
        target_solution[0, 0] = 0
        assert out.sum(axis=1).max() <= 5 and np.array_equal(
            out, target_solution)

    # if input is only ones, will return complying solutions
    def test_check_solution_for_max_work_days_per_week_if_input_is_only_ones_will_return_complying_solutions(
            self,
            check_constraints,
    ):
        solution = np.ones((4, 7), dtype=int)
        out = check_constraints.check_solution_for_max_work_days_per_week(
            solution, 2, 5)
        assert out.sum(axis=1).max() <= 5

    # check_population_for_max_days_per_week
    # returns a ndarray
    def test_check_population_for_max_days_per_week_returns_ndarray(
            self,
            check_constraints,
    ):
        population = np.random.randint(0, 2, size=(5, 4, 7))
        out = check_constraints.check_population_for_max_days_per_week(
            population,
            2,
            5,
        )
        assert type(out) == np.ndarray

    # returns a ndarray or dim (pop_size, nb_nurses, nb_shifts)
    def test_check_population_for_max_days_per_week_returns_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            check_constraints
    ):
        population = np.random.randint(0, 2, size=(5, 4, 7))
        out = check_constraints.check_population_for_max_days_per_week(
            population, 2, 5)
        assert out.shape == (5, 4, 7)

    # each item of output is either 0 or 1
    def test_check_population_for_max_days_per_week_each_item_of_output_is_either_0_or_1(self, check_constraints):
        population = np.random.randint(0, 2, size=(5, 4, 7))
        out = check_constraints.check_population_for_max_days_per_week(
            population, 2, 5)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert out[i, j, k] in [0, 1]

    # each item of output is a solution
    def test_check_population_for_max_days_per_week_each_item_of_output_is_a_solution(self, check_constraints):
        population = np.random.randint(0, 2, size=(5, 4, 7))
        out = check_constraints.check_population_for_max_days_per_week(
            population, 2, 5)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert [self.is_solution(sol) for sol in out]

    # each item of output comply with the max work days per week constraint
    def test_check_population_for_max_days_per_week_each_item_of_output_comply_with_the_max_work_days_per_week_constraint(
            self,
            check_constraints
    ):
        population = np.random.randint(0, 2, size=(5, 4, 7))
        out = check_constraints.check_population_for_max_days_per_week(
            population, 2, 5)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert [sol.sum(axis=1).max() <= 5 for sol in out]

    # if input has a solution not comlying with the max work days per week,
    # removes shifts from overworked nurse
    def test_check_population_for_max_days_per_week_if_input_has_a_solution_not_comlying_with_the_max_work_days_per_week_removes_shifts_from_overworked_nurse(
            self,
            check_constraints
    ):
        population = np.zeros((5, 4, 7), dtype=int)
        for i in range(6):
            population[0, 0, i] = 1
        out = check_constraints.check_population_for_max_days_per_week(
            population, 2, 5)
        assert [sol.sum(axis=1).max() <= 5 for sol in out]

    # if input has a solution not comlying with the max work days per week, and
    # overworked nurse shift can be remove from overcovred shift, it will remove this in priority
    def test_check_population_for_max_days_per_week_if_input_has_a_solution_not_comlying_with_the_max_work_days_per_week_and_overworked_nurse_shift_can_be_remove_from_overcovred_shift_it_will_remove_this_in_priority(
            self,
            check_constraints
    ):
        population = np.zeros((5, 4, 7), dtype=int)
        for i in range(6):
            population[0, 0, i] = 1
        for i in range(1, 4):
            population[0, i, 0] = 1
        out = check_constraints.check_population_for_max_days_per_week(
            population, 2, 5)
        target_population = population.copy()
        target_population[0, 0, 0] = 0
        assert [sol.sum(axis=1).max() <= 5 for sol in out] and np.array_equal(
            out, target_population)

    # if input is only ones, will return complying solutions
    def test_check_population_for_max_days_per_week_if_input_is_only_ones_will_return_complying_solutions(self, check_constraints):
        population = np.ones((5, 4, 7), dtype=int)
        out = check_constraints.check_population_for_max_days_per_week(
            population, 2, 5)
        assert [sol.sum(axis=1).max() <= 5 for sol in out]
