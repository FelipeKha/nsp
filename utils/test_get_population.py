import numpy as np
import pytest

from problem_setup.problem import Problem
from utils.get_population import \
    get_random_nurse_schedule, \
        get_random_initial_solution, \
            get_initial_population

class TestGetPopulation:

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

    @pytest.fixture
    def nurse_schedule(self, problem):
        """
        Returns a random nurse schedule.
        """
        return get_random_nurse_schedule(problem)
    
    @pytest.fixture
    def solution(self, problem):
        """
        Returns a random solution.
        """
        return get_random_initial_solution(problem)
    
    @pytest.fixture
    def population(self, problem):
        """
        Returns a random population of 10 solutions.
        """
        return get_initial_population(10, problem)

    # get_random_nurse_schedule
    def test_get_random_nurse_schedule_returns_ndarray(self, nurse_schedule):
        assert type(nurse_schedule) == np.ndarray

    def test_get_random_nurse_schedule_returns_array_expected_shape(self, nurse_schedule):
        assert nurse_schedule.shape == (7,)

    def test_get_random_nurse_schedule_complies_with_max_workdays(self, nurse_schedule):
        assert nurse_schedule.sum() <= 5
        
    def test_get_random_nurse_schedule_returns_binary_array(self, nurse_schedule):
        result = True
        for i in nurse_schedule:
            if i != 0 and i != 1:
                result = False
        assert result == True


    # get_random_initial_solution
    def test_get_random_initial_solution_returns_ndarray(self, solution):
        assert type(solution) == np.ndarray

    def test_get_random_initial_solution_returns_array_expected_shape(self, solution):
        assert solution.shape == (4, 7)

    def test_get_random_initial_solution_complies_with_max_workdays_for_each_nurse(self, solution):
        assert solution.sum(axis=1).max() <= 5
        
    def test_get_random_initial_solution_returns_binary_array(self, solution):
        result = True
        for i in range(solution.shape[0]):
            for j in range(solution.shape[1]):
                if solution[i][j] != 0 and solution[i][j] != 1:
                    result = False
        assert result == True

    # get_initial_population
    def test_get_initial_population_returns_ndarray(self, population):
        assert type(population) == np.ndarray

    def test_get_initial_population_returns_array_expected_shape(self, population):
        assert population.shape == (10, 4, 7)

    def test_get_initial_population_complies_with_max_workdays_for_each_nurse(self, population):
        assert population.sum(axis=2).max() <= 5
        
    def test_get_initial_population_returns_binary_array(self, population):
        result = True
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                for k in range(population.shape[2]):
                    if population[i][j][k] != 0 and population[i][j][k] != 1:
                        result = False
        assert result == True