import numpy as np
import pytest

from utils.covering_cost import CoveringCost
from utils.get_neighbour import GetNeighbour
from utils.get_population import GetPopulation


class TestGetNeighbour:
    @pytest.fixture
    def get_neighbour(self):
        """
        Returns a GetNeighbour instance with:
        - Nb nurses: 4
        - Work days: 7
        - Shift per workday: 1
        - Max work days per week: 5
        - CoveringCost instance
        """
        return GetNeighbour(
            nb_nurses=4,
            nb_work_days_per_week=7,
            nb_shifts_per_work_day=1,
            nrs_max_work_days_per_week=5,
            covering_cost=CoveringCost(
                nb_work_days_per_week=7,
                nb_shifts_per_work_day=1,
                nb_nrs_per_shift=2,
            ),
        )

    @pytest.fixture
    def random_solution(self):
        """
        Returns a random solution.
        """
        get_population = GetPopulation(
            nb_nurses=4,
            nb_work_days_per_week=7,
            nb_shifts_per_work_day=1,
            nrs_max_work_days_per_week=5,
        )
        return get_population.get_random_initial_solution()

    @pytest.fixture
    def covering_cost(self):
        """
        Returns a covering cost instance.
        """
        return CoveringCost(
            nb_work_days_per_week=7,
            nb_shifts_per_work_day=1,
            nb_nrs_per_shift=2,
        )

    def is_solution(self, solution):
        """
        Returns True if solution is a solution. Being a solution means:
        - solution is an ndarray
        - solution has the expected shape (nb_nurses, nb_work_days_per_week * 
        nb_shifts_per_work_day)
        - solution is binary
        - solution comply with the max work days per week constraint
        """
        if type(solution) != np.ndarray:
            return False
        if solution.shape != (4, 7):
            return False
        if not np.array_equal(solution, solution.astype(bool)):
            return False
        if solution.sum(axis=1).max() > 5:
            return False
        return True

    def is_neighbour(self, neighbour, solution):
        """
        Returns True if neighbour is a neighbour of solution. Being a neighbour 
        means that for a random nurse, a random shift is changed. All other 
        shifts are unchanged.
        """
        neighbour += np.ones_like(neighbour)
        solution += np.ones_like(solution)
        diff = neighbour - solution
        diff = np.square(diff)
        if diff.sum() != 1:
            return False
        return True

    def is_tabu_list(self, tabu_list):
        """
        Returns True if tabu_list is a tabu list. Being a tabu list means:
        - tabu_list is a dictionary
        - tabu_list keys are solutions
        - tabu_list values are integers
        """
        if type(tabu_list) != dict:
            return False
        for key in tabu_list:
            if not self.is_solution(key):
                return False
            if type(tabu_list[key]) != int:
                return False
        return True

    # get_neighbour
    # return tuple of 2 elements
    def test_get_neighbour_returns_tuple(self, get_neighbour, random_solution):
        out = get_neighbour.get_neighbour(random_solution, 2)
        assert type(out) == tuple and len(out) == 2

    # first element is of type ndarray
    def test_get_neighbour_returns_ndarray(
        self,
        get_neighbour,
        random_solution
    ):
        out = get_neighbour.get_neighbour(random_solution, 2)
        assert type(out[0]) == np.ndarray

    # second element is of type np.int64
    def test_get_neighbour_returns_np_int64(
        self,
        get_neighbour,
        random_solution
    ):
        out = get_neighbour.get_neighbour(random_solution, 2)
        assert type(out[1]) == np.int64

    # first element is a solution
    def test_get_neighbour_returns_solution(
        self,
        get_neighbour,
        random_solution
    ):
        out = get_neighbour.get_neighbour(random_solution, 2)
        assert self.is_solution(out[0])

    # first element is different from the input solution
    def test_get_neighbour_returns_different_solution(
        self,
        get_neighbour,
        random_solution
    ):
        out = get_neighbour.get_neighbour(random_solution, 2)
        assert not np.array_equal(out[0], random_solution)

    # fist element is a neighbour of the input solution
    def test_get_neighbour_returns_neighbour(
        self,
        get_neighbour,
        random_solution
    ):
        out = get_neighbour.get_neighbour(random_solution, 2)
        assert self.is_neighbour(out[0], random_solution)

    # first element comply with the max work days per week constraint
    def test_get_neighbour_returns_solution_comply_max_work_days_per_week(
        self,
        get_neighbour,
        random_solution
    ):
        out = get_neighbour.get_neighbour(random_solution, 2)
        assert out[0].sum(axis=1).max() <= 5

    # second element is the covering cost of the first element
    def test_get_neighbour_returns_covering_cost(
        self,
        get_neighbour,
        random_solution,
        covering_cost
    ):
        out = get_neighbour.get_neighbour(random_solution, 2)
        assert out[1] == covering_cost.covering_cost(out[0])

    # get_neighbour_tabu
    # return tuple of 3 elements
    def test_get_neighbour_tabu_returns_tuple(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert type(out) == tuple and len(out) == 3

    # first element is of type ndarray
    def test_get_neighbour_tabu_returns_ndarray(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert type(out[0]) == np.ndarray

    # second element is of type np.int64
    def test_get_neighbour_tabu_returns_np_float64(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert type(out[1]) == np.int64

    # third element is of type dict
    def test_get_neighbour_tabu_returns_dict(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert type(out[2]) == dict

    # first element is a solution
    def test_get_neighbour_tabu_returns_solution(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert self.is_solution(out[0])

    # first element is different from the input solution
    def test_get_neighbour_tabu_returns_different_solution(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert not np.array_equal(out[0], random_solution)

    # fist element is a neighbour of the input solution
    def test_get_neighbour_tabu_returns_neighbour(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert self.is_neighbour(out[0], random_solution)

    # first element comply with the max work days per week constraint
    def test_get_neighbour_tabu_returns_solution_comply_max_work_days_per_week(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert out[0].sum(axis=1).max() <= 5

    # second element is the covering cost of the first element
    def test_get_neighbour_tabu_returns_covering_cost(
            self,
            get_neighbour,
            random_solution,
            covering_cost
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert out[1] == covering_cost.covering_cost(out[0])

    # third element is a tabu list
    def test_get_neighbour_tabu_returns_tabu_list(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert type(out[2]) == dict

    # third element is a tabu list with the input solution as key
    def test_get_neighbour_tabu_returns_solution_not_in_tabu_list(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert tuple(map(tuple, random_solution)) in out[2]

    # third element is a tabu list with the input solution as key and the value
    # is 5
    def test_get_neighbour_tabu_returns_tabu_list_with_updated_input_tabu_list(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert out[2][tuple(map(tuple, random_solution))] == 5

    # third element is a tabu list with the first element as key
    def test_get_neighbour_tabu_returns_tabu_list_with_first_element_as_key(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert tuple(map(tuple, out[0])) in out[2]

    # third element is a tabu list with the first element as key and the value
    # is 10
    def test_get_neighbour_tabu_returns_tl_with_first_element_key_and_value_10(
            self,
            get_neighbour,
            random_solution
    ):
        tabu_list = {tuple(map(tuple, random_solution)): 5}
        out = get_neighbour.get_neighbour_tabu(
            random_solution, 2, tabu_list, 10)
        assert out[2][tuple(map(tuple, out[0]))] == 10
