import numpy as np
import pytest

from problem_setup.problem import Problem
from tabu_search.tabu_search import TabuSearch
from utils.covering_cost import covering_cost
from utils.get_neighbour import get_neighbour_tabu
from utils.get_population import get_random_initial_solution


class TestTabuSearch:
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
    def tabu_search(self):
        return TabuSearch(
            nb_iter=10,
            nb_neighbours=2,
            tabu_limit=10,
            get_random_initial_solution=get_random_initial_solution,
            get_neighbour_tabu=get_neighbour_tabu,
            covering_cost=covering_cost,
        )

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

    # tabu_search
    # return tuple of len 3
    def test_tabu_search_return_tuple_of_len_3(self, tabu_search, problem):
        out = tabu_search(problem)
        assert len(out) == 3

    # first element is a numpy array
    def test_tabu_search_first_element_is_a_numpy_array(
            self,
            tabu_search,
            problem,
    ):
        out = tabu_search(problem)
        assert type(out[0]) == np.ndarray

    # first element is a solution
    def test_tabu_search_first_element_is_a_solution(
            self,
            tabu_search,
            problem
    ):
        out = tabu_search(problem)
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_tabu_search_first_element_comply_max_work_days_per_week(
            self,
            tabu_search,
            problem,
    ):
        out = tabu_search(problem)
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_tabu_search_second_element_is_np_int64(self, tabu_search, problem):
        out = tabu_search(problem)
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_tabu_search_second_element_is_covering_cost_of_first_element(
            self,
            tabu_search,
            problem,
    ):
        out = tabu_search(problem)
        assert out[1] == covering_cost(out[0], problem)

    # third element is a list
    def test_tabu_search_third_element_is_a_list(self, tabu_search, problem):
        out = tabu_search(problem)
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_tabu_search_third_element_is_a_list_of_np_int64(
            self,
            tabu_search,
            problem,
    ):
        out = tabu_search(problem)
        print('type out[2][0]', type(out[2][0]))
        for i in out[2]:
            print('type:', type(i), 'value:', i)
        assert all(type(x) == np.int64 or x == np.inf for x in out[2])

    # third element items are all greater than or equal to 0
    def test_tabu_search_third_element_items_are_all_greater_than_or_equal_to_0(
            self,
            tabu_search,
            problem,
    ):
        out = tabu_search(problem)
        assert all(x >= 0 for x in out[2])

    # __call__
    # return tuple of len 3
    def test_search_solution_return_tuple_of_len_3(self, tabu_search, problem):
        out = tabu_search(problem)
        assert len(out) == 3

    # first element is a numpy array
    def test_search_solution_first_element_is_a_numpy_array(self, tabu_search, problem):
        out = tabu_search(problem)
        assert type(out[0]) == np.ndarray

    # first element is a solution
    def test_search_solution_first_element_is_a_solution(self, tabu_search, problem):
        out = tabu_search(problem)
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_search_solution_first_element_comply_max_work_days_per_week(
            self,
            tabu_search,
            problem,
    ):
        out = tabu_search(problem)
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_search_solution_second_element_is_np_int64(self, tabu_search, problem):
        out = tabu_search(problem)
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_search_solution_second_element_is_covering_cost_of_first_element(
            self,
            tabu_search,
            problem,
    ):
        out = tabu_search(problem)
        assert out[1] == covering_cost(out[0], problem)

    # third element is a list
    def test_search_solution_third_element_is_a_list(self, tabu_search, problem):
        out = tabu_search(problem)
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_search_solution_third_element_is_a_list_of_np_int64(
            self,
            tabu_search, 
            problem,
    ):
        out = tabu_search(problem)
        print('type out[2][0]', type(out[2][0]))
        for i in out[2]:
            print('type:', type(i), 'value:', i)
        assert all(type(x) == np.int64 or x == np.inf for x in out[2])

    # third element items are all greater than or equal to 0
    def test_search_solution_third_element_items_are_greater_than_or_equal_to_0(
            self,
            tabu_search, 
            problem,
    ):
        out = tabu_search(problem)
        assert all(x >= 0 for x in out[2])