import numpy as np
import pytest

from problem_setup.problem import Problem
from simulated_annealing.sim_anneal import SimulatedAnnealing
from utils.covering_cost import covering_cost
from utils.get_neighbour import get_neighbour
from utils.get_population import get_random_initial_solution


class TestSimulatedAnnealing:
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
    def simulated_annealing(self):
        return SimulatedAnnealing(
            nb_iter=20,
            nb_neighbours=2,
            k=20,
            lam=0.005,
            limit=10,
            get_random_initial_solution=get_random_initial_solution,
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
        """
        if type(solution) != np.ndarray:
            return False
        if solution.shape != (4, 7):
            return False
        if not np.array_equal(solution, solution.astype(bool)):
            return False
        return True

    # exp_schedule
    # return a np.float64
    def test_exp_schedule_return_a_np_float64(self, simulated_annealing):
        t = 1
        assert type(simulated_annealing.exp_schedule(
            t,
            simulated_annealing.k,
            simulated_annealing.lam,
            simulated_annealing.limit,
        )) == np.float64

    # return k * np.exp(-lam * t) if t < limit
    def test_exp_schedule_return_k_exp_minus_lam_t_if_t_less_than_limit(
        self,
        simulated_annealing,
    ):
        t = simulated_annealing.limit - 1
        assert simulated_annealing.exp_schedule(
            t,
            simulated_annealing.k,
            simulated_annealing.lam,
            simulated_annealing.limit,
        ) == simulated_annealing.k * np.exp(-simulated_annealing.lam * t)

    # return 0 if t >= limit
    def test_exp_schedule_return_0_if_t_greater_than_or_equal_to_limit(
        self,
        simulated_annealing,
    ):
        t = simulated_annealing.limit + 1
        assert simulated_annealing.exp_schedule(
            t,
            simulated_annealing.k,
            simulated_annealing.lam,
            simulated_annealing.limit,
        ) == 0

    # simulated_annealing
    # return tuple of len 3
    def test_simulated_annealing_return_tuple_of_len_3(
            self,
            simulated_annealing,
            problem,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        assert len(out) == 3

    # first element is a numpy array
    def test_simulated_annealing_first_element_is_a_numpy_array(self, simulated_annealing, problem):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        assert type(out[0]) == np.ndarray

    # first element is a solution
    def test_simulated_annealing_first_element_is_a_solution(self, simulated_annealing, problem):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_simulated_annealing_first_element_comply_with_the_max_work_days_per_week_constraint(
        self,
        simulated_annealing,
        problem,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_simulated_annealing_second_element_is_of_type_np_int64(self, simulated_annealing, problem):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_simulated_annealing_second_element_is_the_covering_cost_of_the_first_element(
        self,
        simulated_annealing,
        problem,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        assert out[1] == covering_cost(out[0], problem)

    # covering cost of first element is less than or equal to the covering cost
    # of the initial solution
    def test_simulated_annealing_cov_cost_of_first_el_less_than_or_equal_to_cov_cost_of_init_sol(
        self,
        simulated_annealing,
        problem,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        assert out[1] <= covering_cost(initial_solution, problem)

    # third element is a list
    def test_simulated_annealing_third_element_is_a_list(self, simulated_annealing, problem):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_simulated_annealing_third_element_is_a_list_of_np_int64(
            self,
            simulated_annealing,
            problem
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        print('type out[2][0]', type(out[2][0]))
        for i in out[2]:
            print('type:', type(i), 'value:', i)
        assert all(type(x) == np.int64 or x == np.inf for x in out[2])

    # third element items are all greater than or equal to 0
    def test_simulated_annealing_third_element_items_are_all_greater_than_or_equal_to_0(
        self,
        simulated_annealing,
        problem,
    ):
        initial_solution = np.zeros((4, 7))
        out = simulated_annealing.simulated_annealing(initial_solution, problem)
        assert all([item >= 0 for item in out[2]])

    # __call__
    # return tuple of len 3
    def test_call_return_tuple_of_len_3(
        self,
        simulated_annealing,
        problem,
    ):
        out = simulated_annealing(problem)
        assert len(out) == 3

    # first element is a numpy array
    def test_call_first_element_is_a_numpy_array(self, simulated_annealing, problem):
        out = simulated_annealing(problem)
        assert type(out[0]) == np.ndarray

    # first element is a solution
    def test_call_first_element_is_a_solution(self, simulated_annealing, problem):
        out = simulated_annealing(problem)
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_call_first_element_comply_with_the_max_work_days_per_week_constraint(
        self,
        simulated_annealing,
        problem,
    ):
        out = simulated_annealing(problem)
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_call_second_element_is_of_type_np_int64(self, simulated_annealing, problem):
        out = simulated_annealing(problem)
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_call_second_element_is_the_covering_cost_of_the_first_element(
        self,
        simulated_annealing,
        problem,
    ):
        out = simulated_annealing(problem)
        assert out[1] == covering_cost(out[0], problem)

    # third element is a list
    def test_call_third_element_is_a_list(self, simulated_annealing, problem):
        out = simulated_annealing(problem)
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_call_third_element_is_a_list_of_np_int64(
            self,
            simulated_annealing,
            problem,
    ):
        out = simulated_annealing(problem)
        print('type out[2][0]', type(out[2][0]))
        for i in out[2]:
            print('type:', type(i), 'value:', i)
        assert all(type(x) == np.int64 or x == np.inf for x in out[2])

    # third element items are all greater than or equal to 0
    def test_call_third_element_items_are_all_greater_than_or_equal_to_0(
        self,
        simulated_annealing,
        problem,
    ):
        out = simulated_annealing(problem)
        assert all([item >= 0 for item in out[2]])
