import numpy as np
import pytest

from simulated_annealing.sim_anneal import SimulatedAnnealing
from utils.covering_cost import CoveringCost
from utils.get_neighbour import GetNeighbour
from utils.get_population import GetPopulation


class TestSimulatedAnnealing:
    @pytest.fixture
    def get_population(self):
        """
        Returns a GetPopulation instance with:
        - Work days: 7
        - Shift per workday: 1
        - Max work days per week: 5
        """
        return GetPopulation(
            nb_nurses=4,
            nb_work_days_per_week=7,
            nb_shifts_per_work_day=1,
            nrs_max_work_days_per_week=5,
        )

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

    @pytest.fixture
    def simulated_annealing(self, get_population, get_neighbour, covering_cost):
        return SimulatedAnnealing(
            nb_nurses=4,
            nb_work_days_per_week=7,
            nb_shifts_per_work_day=1,
            nb_nrs_per_shift=2,
            nrs_max_work_days_per_week=5,
            nb_iter=20,
            nb_neighbours=2,
            k=20,
            lam=0.005,
            limit=10,
            get_population=get_population,
            get_neighbour=get_neighbour,
            covering_cost=covering_cost,
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

    # simulated_annealing
    # return tuple of len 3
    def test_simulated_annealing_return_tuple_of_len_3(
            self,
            simulated_annealing,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        assert len(out) == 3

    # first element is a numpy array
    def test_first_element_is_a_numpy_array(self, simulated_annealing):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        assert type(out[0]) == np.ndarray

    # first element is a solution
    def test_first_element_is_a_solution(self, simulated_annealing):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_first_element_comply_with_the_max_work_days_per_week_constraint(
        self,
        simulated_annealing,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_second_element_is_of_type_np_int64(self, simulated_annealing):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_second_element_is_the_covering_cost_of_the_first_element(
        self,
        simulated_annealing,
        covering_cost,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        assert out[1] == covering_cost.covering_cost(out[0])

    # covering cost of first element is less than or equal to the covering cost
    # of the initial solution
    def test_cov_cost_of_first_el_less_than_or_equal_to_cov_cost_of_init_sol(
        self,
        simulated_annealing,
        covering_cost,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        assert out[1] <= covering_cost.covering_cost(initial_solution)

    # third element is a list
    def test_third_element_is_a_list(self, simulated_annealing):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_third_element_is_a_list_of_np_int64(
            self,
            simulated_annealing
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        print('type out[2][0]', type(out[2][0]))
        for i in out[2]:
            print('type:', type(i), 'value:', i)
        assert all(type(x) == np.int64 or x == np.inf for x in out[2])

    # third element items are all greater than or equal to 0
    def test_third_element_items_are_all_greater_than_or_equal_to_0(
        self,
        simulated_annealing,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution)
        assert all([item >= 0 for item in out[2]])

    # search_solution
    # return tuple of len 3
    def test_search_solution_return_tuple_of_len_3(
        self,
        simulated_annealing,
    ):
        out = simulated_annealing.search_solution()
        assert len(out) == 3

    # first element is a numpy array
    def test_first_element_is_a_numpy_array(self, simulated_annealing):
        out = simulated_annealing.search_solution()
        assert type(out[0]) == np.ndarray

    # first element is a solution
    def test_first_element_is_a_solution(self, simulated_annealing):
        out = simulated_annealing.search_solution()
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_first_element_comply_with_the_max_work_days_per_week_constraint(
        self,
        simulated_annealing,
    ):
        out = simulated_annealing.search_solution()
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_second_element_is_of_type_np_int64(self, simulated_annealing):
        out = simulated_annealing.search_solution()
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_second_element_is_the_covering_cost_of_the_first_element(
        self,
        simulated_annealing,
        covering_cost,
    ):
        out = simulated_annealing.search_solution()
        assert out[1] == covering_cost.covering_cost(out[0])

    # third element is a list
    def test_third_element_is_a_list(self, simulated_annealing):
        out = simulated_annealing.search_solution()
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_third_element_is_a_list_of_np_int64(
            self,
            simulated_annealing
    ):
        out = simulated_annealing.search_solution()
        print('type out[2][0]', type(out[2][0]))
        for i in out[2]:
            print('type:', type(i), 'value:', i)
        assert all(type(x) == np.int64 or x == np.inf for x in out[2])

    # third element items are all greater than or equal to 0
    def test_third_element_items_are_all_greater_than_or_equal_to_0(
        self,
        simulated_annealing,
    ):
        out = simulated_annealing.search_solution()
        assert all([item >= 0 for item in out[2]])
