import numpy as np
import random
import pytest

from particle_swarm.pso import ParticleSwarmOptimization
from utils.covering_cost import CoveringCost
from utils.get_neighbour import GetNeighbour
from utils.get_population import GetPopulation


class TestParticleSwarmOptimization:
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
    def part_swarm(self, get_population, covering_cost):
        return ParticleSwarmOptimization(
            nb_nurses=4,
            nb_work_days_per_week=7,
            nb_shifts_per_work_day=1,
            nb_nrs_per_shift=2,
            nrs_max_work_days_per_week=5,
            swarm_size=5,
            max_iter=20,
            c1=0.7,
            c2=0.3,
            w=0.75,
            alpha=0.3,
            get_population=get_population,
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

    # get_pop_costs
    # return a ndarray
    def test_get_pop_costs_returns_ndarray(self, part_swarm):
        swarm = np.random.randint(2, size=(5, 4, 7))
        out = part_swarm.get_pop_costs(swarm)
        assert type(out) == np.ndarray

    # returns a ndarray of shape (swarm_size)
    def test_get_pop_costs_returns_ndarray_of_shape_swarm_size(
            self,
            part_swarm
    ):
        swarm = np.random.randint(2, size=(5, 4, 7))
        out = part_swarm.get_pop_costs(swarm)
        assert out.shape == (5,)

    # each item of output is covering cost of input
    def test_get_pop_costs_each_item_of_output_is_covering_cost_of_input(
            self,
            part_swarm,
            covering_cost
    ):
        swarm = np.random.randint(2, size=(5, 4, 7))
        out = part_swarm.get_pop_costs(swarm)
        for i in range(5):
            assert out[i] == covering_cost.covering_cost(swarm[i])

    # each item of outputs is greater than or equal to 0
    def test_get_pop_costs_each_item_of_outputs_is_greater_than_or_equal_to_0(
            self,
            part_swarm
    ):
        swarm = np.random.randint(2, size=(5, 4, 7))
        out = part_swarm.get_pop_costs(swarm)
        assert (out >= 0).all()

    # ycompare_func
    # returns an np.int64
    def test_ycompare_func_returns_np_int64(self, part_swarm):
        y = random.randint(0, 1)
        pbest = random.randint(0, 1)
        gbest = random.randint(0, 1)
        out = part_swarm.ycompare_func(y, pbest, gbest)
        assert type(out) == np.int64

    # returns 1 if y = gbest (current shift same as swarm best)
    def test_ycompare_func_returns_1_if_y_equals_gbest(
            self,
            part_swarm
    ):
        y = random.randint(0, 1)
        pbest = 1 - y
        gbest = y
        out = part_swarm.ycompare_func(y, pbest, gbest)
        assert out == 1

    # returns -1 if y = pbest (current shift same as personal best)
    def test_ycompare_func_returns_minus_1_if_y_equals_pbest(
            self,
            part_swarm
    ):
        y = random.randint(0, 1)
        pbest = y
        gbest = 1 - y
        out = part_swarm.ycompare_func(y, pbest, gbest)
        assert out == -1

    # returns -1 or 1 randomly if y = gbest = pbest
    def test_ycompare_func_returns_minus_1_or_1_randomly_if_y_equals_gbest_equals_pbest(
            self,
            part_swarm
    ):
        y = random.randint(0, 1)
        pbest = y
        gbest = y
        out = part_swarm.ycompare_func(y, pbest, gbest)
        assert out in [-1, 1]

    # returns 0 otherwise
    def test_ycompare_func_returns_0_otherwise(self, part_swarm):
        y = random.randint(0, 1)
        pbest = 1 - y
        gbest = 1 - y
        out = part_swarm.ycompare_func(y, pbest, gbest)
        assert out == 0

    # yupdate_func
    # returns an int
    def test_yupdate_func_returns_int(self, part_swarm):
        ylambda = random.random()
        out = part_swarm.yupdate_func(ylambda)
        assert type(out) == int

    # returns 1 if ylambda > alpha
    def test_yupdate_func_returns_1_if_ylambda_greater_than_alpha(
            self,
            part_swarm
    ):
        ylambda = part_swarm.alpha + 0.1
        out = part_swarm.yupdate_func(ylambda)
        assert out == 1

    # returns -1 if ylambda < alpha
    def test_yupdate_func_returns_minus_1_if_ylambda_less_than_alpha(
            self,
            part_swarm
    ):
        ylambda = part_swarm.alpha - 0.1
        out = part_swarm.yupdate_func(ylambda)
        assert out == -1

    # returns 0 otherwise
    def test_yupdate_func_returns_0_otherwise(self, part_swarm):
        ylambda = part_swarm.alpha
        out = part_swarm.yupdate_func(ylambda)
        assert out == 0

    # swarm_update_func
    # returns an np.int64
    def test_swarm_update_func_returns_np_int64(self, part_swarm):
        y = random.randint(0, 1)
        pb = random.randint(0, 1)
        gb = random.randint(0, 1)
        out = part_swarm.swarm_update_func(y, pb, gb)
        assert type(out) == np.int64

    # returns pb if y = -1
    def test_swarm_update_func_returns_pb_if_y_equals_minus_1(
            self,
            part_swarm
    ):
        y = -1
        pb = random.randint(0, 1)
        gb = 1 - pb
        out = part_swarm.swarm_update_func(y, pb, gb)
        assert out == pb

    # returns gb if y = 1
    def test_swarm_update_func_returns_gb_if_y_equals_1(
            self,
            part_swarm
    ):
        y = 1
        pb = random.randint(0, 1)
        gb = 1 - pb
        out = part_swarm.swarm_update_func(y, pb, gb)
        assert out == gb
    # returns 0 or 1 randomly if y = 0

    def test_swarm_update_func_returns_0_or_1_randomly_if_y_equals_0(
            self,
            part_swarm
    ):
        y = 0
        pb = random.randint(0, 1)
        gb = random.randint(0, 1)
        out = part_swarm.swarm_update_func(y, pb, gb)
        assert out in [0, 1]

    # get_ycompare
    # returns a ndarray
    def test_get_ycompare_returns_ndarray(self, part_swarm):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        gbest = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.get_ycompare(pcurrent, pbest, gbest)
        assert type(out) == np.ndarray

    # returns a ndarray of dim (swarm_size, nb_nurses, nb_shifts)
    def test_get_ycompare_returns_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            part_swarm
    ):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        gbest = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.get_ycompare(pcurrent, pbest, gbest)
        assert out.shape == (5, 4, 7)

    # apply ycompare_func to each item of pcurrent input
    def test_get_ycompare_apply_ycompare_func_to_each_item_of_pcurrent_input(
            self,
            part_swarm
    ):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        gbest = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.get_ycompare(pcurrent, pbest, gbest)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    if pcurrent[i, j, k] == pbest[i, j, k] and pcurrent[i, j, k] == gbest[i, j, k]:
                        assert out[i, j, k] in [-1, 1]
                    else:
                        assert out[i, j, k] == part_swarm.ycompare_func(
                            pcurrent[i, j, k],
                            pbest[i, j, k],
                            gbest[i, j, k],
                        )

    # each item of output is either -1, 0 or 1
    def test_get_ycompare_each_item_of_output_is_either_minus_1_0_or_1(
            self,
            part_swarm
    ):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        gbest = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.get_ycompare(pcurrent, pbest, gbest)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert out[i, j, k] in [-1, 0, 1]

    # get_yupdate
    # returns a ndarray
    def test_get_yupdate_returns_ndarray(self, part_swarm):
        ylambda = np.random.rand(5, 4, 7)
        out = part_swarm.get_yupdate(ylambda)
        assert type(out) == np.ndarray

    # returns a ndarray of dim (swarm_size, nb_nurses, nb_shifts)
    def test_get_yupdate_returns_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            part_swarm
    ):
        ylambda = np.random.rand(5, 4, 7)
        out = part_swarm.get_yupdate(ylambda)
        assert out.shape == (5, 4, 7)

    # apply yupdate_func to each item of ylambda input
    def test_get_yupdate_apply_yupdate_func_to_each_item_of_ylambda_input(
            self,
            part_swarm
    ):
        ylambda = np.random.rand(5, 4, 7)
        out = part_swarm.get_yupdate(ylambda)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert out[i, j, k] == part_swarm.yupdate_func(
                        ylambda[i, j, k])

    # each item of output is either -1, 0 or 1
    def test_get_yupdate_each_item_of_output_is_either_minus_1_0_or_1(
            self,
            part_swarm
    ):
        ylambda = np.random.rand(5, 4, 7)
        out = part_swarm.get_yupdate(ylambda)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert out[i, j, k] in [-1, 0, 1]

    # get_d1_d2
    # returns a tuple of len 2
    def test_get_d1_d2_returns_tuple_of_len_2(self, part_swarm):
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        out = part_swarm.get_d1_d2(ycompare)
        assert len(out) == 2

    # first element is a ndarray
    def test_get_d1_d2_first_element_is_ndarray(self, part_swarm):
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        out = part_swarm.get_d1_d2(ycompare)
        assert type(out[0]) == np.ndarray

    # first element is a ndarray of dim (swarm_size, nb_nurses, nb_shifts)
    def test_get_d1_d2_first_element_is_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            part_swarm
    ):
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        out = part_swarm.get_d1_d2(ycompare)
        assert out[0].shape == (5, 4, 7)

    # first element is -1 - ycompare
    def test_get_d1_d2_first_element_is_minus_1_minus_ycompare(
            self,
            part_swarm
    ):
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        out = part_swarm.get_d1_d2(ycompare)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert out[0][i, j, k] == -1 - ycompare[i, j, k]

    # first element values are either -2, -1 or 0
    def test_get_d1_d2_first_element_values_are_either_minus_2_minus_1_or_0(
            self,
            part_swarm
    ):
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        out = part_swarm.get_d1_d2(ycompare)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert out[0][i, j, k] in [-2, -1, 0]

    # second element is a ndarray
    def test_get_d1_d2_second_element_is_ndarray(self, part_swarm):
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        out = part_swarm.get_d1_d2(ycompare)
        assert type(out[1]) == np.ndarray

    # second element is a ndarray of dim (swarm_size, nb_nurses, nb_shifts)
    def test_get_d1_d2_second_element_is_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            part_swarm
    ):
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        out = part_swarm.get_d1_d2(ycompare)
        assert out[1].shape == (5, 4, 7)

    # second element is 1 - ycompare
    def test_get_d1_d2_second_element_is_1_minus_ycompare(
            self,
            part_swarm
    ):
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        out = part_swarm.get_d1_d2(ycompare)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert out[1][i, j, k] == 1 - ycompare[i, j, k]

    # second element values are either 0, 1 or 2
    def test_get_d1_d2_second_element_values_are_either_0_1_or_2(
            self,
            part_swarm
    ):
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        out = part_swarm.get_d1_d2(ycompare)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert out[1][i, j, k] in [0, 1, 2]

    # check_swarm
    # returns a ndarray
    def test_check_swarm_returns_ndarray(self, part_swarm):
        swarm = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.check_swarm(swarm)
        assert type(out) == np.ndarray

    # returns a ndarray or dim (swarm_size, nb_nurses, nb_shifts)
    def test_check_swarm_returns_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            part_swarm
    ):
        swarm = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.check_swarm(swarm)
        assert out.shape == (5, 4, 7)

    # each item of output is either 0 or 1
    def test_check_swarm_each_item_of_output_is_either_0_or_1(self, part_swarm):
        swarm = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.check_swarm(swarm)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert out[i, j, k] in [0, 1]

    # each item of output is a solution
    def test_check_swarm_each_item_of_output_is_a_solution(self, part_swarm):
        swarm = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.check_swarm(swarm)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert [self.is_solution(sol) for sol in out]

    # each item of output comply with the max work days per week constraint
    def test_check_swarm_each_item_of_output_comply_with_the_max_work_days_per_week_constraint(
            self,
            part_swarm
    ):
        swarm = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.check_swarm(swarm)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    assert [sol.sum(axis=1).max() <= 5 for sol in out]

    # if input has a solution not comlying with the max work days per week,
    # removes shifts from overworked nurse
    def test_check_swarm_if_input_has_a_solution_not_comlying_with_the_max_work_days_per_week_removes_shifts_from_overworked_nurse(
            self,
            part_swarm
    ):
        swarm = np.zeros((5, 4, 7), dtype=int)
        for i in range(6):
            swarm[0, 0, i] = 1
        out = part_swarm.check_swarm(swarm)
        assert [sol.sum(axis=1).max() <= 5 for sol in out]

    # if inout has a solution not comlying with the max work days per week, and
    # overworked nurse shift can be remove from overcovred shift, it will remove this in priority
    def test_check_swarm_if_input_has_a_solution_not_comlying_with_the_max_work_days_per_week_and_overworked_nurse_shift_can_be_remove_from_overcovred_shift_it_will_remove_this_in_priority(
            self,
            part_swarm
    ):
        swarm = np.zeros((5, 4, 7), dtype=int)
        for i in range(6):
            swarm[0, 0, i] = 1
        for i in range(1, 4):
            swarm[0, i, 0] = 1
        out = part_swarm.check_swarm(swarm)
        target_swarm = swarm.copy()
        target_swarm[0, 0, 0] = 0
        assert [sol.sum(axis=1).max() <= 5 for sol in out] and np.array_equal(
            out, target_swarm)

    # update_swarm
    # returns a ndarray
    def test_update_swarm_returns_ndarray(self, part_swarm):
        yupdate = np.random.randint(-1, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        gbest = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.update_swarm(yupdate, pbest, gbest)
        assert type(out) == np.ndarray

    # returns a ndarray of dim (swarm_size, nb_nurses, nb_shifts)
    def test_update_swarm_returns_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            part_swarm
    ):
        yupdate = np.random.randint(-1, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        gbest = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.update_swarm(yupdate, pbest, gbest)
        assert out.shape == (5, 4, 7)

    # apply swarm_update_func to each item of yupdate input
    def test_update_swarm_apply_swarm_update_func_to_each_item_of_yupdate_input(
            self,
            part_swarm
    ):
        yupdate = np.random.randint(-1, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        gbest = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.update_swarm(yupdate, pbest, gbest)
        for i in range(5):
            for j in range(4):
                for k in range(7):
                    if yupdate[i, j, k] == 0:
                        assert out[i, j, k] in [0, 1]
                    else:
                        assert out[i, j, k] == part_swarm.swarm_update_func(
                            yupdate[i, j, k], pbest[i, j, k], gbest[i, j, k])

    # each item of output is either 0 or 1
    def test_update_swarm_each_item_of_output_is_either_0_or_1(self, part_swarm):
        yupdate = np.random.randint(-1, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        gbest = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.update_swarm(yupdate, pbest, gbest)
        assert [np.all((out == 0) | (out == 1))]

    # each item of output is a solution
    def test_update_swarm_each_item_of_output_is_a_solution(self, part_swarm):
        yupdate = np.random.randint(-1, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        gbest = np.random.randint(0, 2, size=(5, 4, 7))
        out = part_swarm.update_swarm(yupdate, pbest, gbest)
        assert [self.is_solution(sol) for sol in out]

    # update_pbest
    # returns a tuple of len 2
    def test_update_pbest_returns_tuple_of_len_2(self, part_swarm):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        assert len(out) == 2

    # first element is a ndarray
    def test_update_pbest_first_element_is_ndarray(self, part_swarm):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        assert type(out[0]) == np.ndarray

    # first element is a ndarray of dim (swarm_size, nb_nurses, nb_shifts)
    def test_update_pbest_first_element_is_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            part_swarm
    ):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        assert out[0].shape == (5, 4, 7)

    # first element is either 1 or 0
    def test_update_pbest_first_element_is_either_1_or_0(self, part_swarm):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        assert np.all((out[0] == 0) | (out[0] == 1))

    # first element is a solution
    def test_update_pbest_first_element_is_a_solution(self, part_swarm):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        assert [self.is_solution(sol) for sol in out[0]]

    # if pcurrent is a solution with a lower covering cost than pbest, pbest is
    # updated with pcurrent
    def test_update_pbest_if_pcurrent_is_a_solution_with_a_lower_covering_cost_than_pbest_pbest_is_updated_with_pcurrent(
            self,
            part_swarm
    ):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.zeros((5, 4, 7), dtype=int)
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        assert np.array_equal(out[0], pcurrent)

    # second element is a ndarray
    def test_update_pbest_second_element_is_ndarray(self, part_swarm):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        assert type(out[1]) == np.ndarray

    # second element is a ndarray of dim (swarm_size)
    def test_update_pbest_second_element_is_ndarray_of_dim_swarm_size(
            self,
            part_swarm
    ):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        assert out[1].shape == (5,)

    # second element is covering cost of first element
    def test_update_pbest_second_element_is_covering_cost_of_first_element(
            self,
            part_swarm,
            covering_cost,
    ):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        target_out = part_swarm.get_pop_costs(out[0])
        assert np.array_equal(out[1], target_out)
    # second element is greater than or equal to 0

    def test_update_pbest_second_element_is_greater_than_or_equal_to_0(
            self,
            part_swarm
    ):
        pcurrent = np.random.randint(0, 2, size=(5, 4, 7))
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pcurrent_costs = part_swarm.get_pop_costs(pcurrent)
        pbest_costs = part_swarm.get_pop_costs(pbest)
        out = part_swarm.update_pbest(
            pcurrent,
            pcurrent_costs,
            pbest,
            pbest_costs
        )
        assert np.all(out[1] >= 0)

    # update_gbest
    # returns a tuple of len 2
    def test_update_gbest_returns_tuple_of_len_2(
            self,
            part_swarm,
            covering_cost
    ):
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pbest_costs = part_swarm.get_pop_costs(pbest)
        gbest = np.random.randint(0, 2, size=(4, 7))
        gbest_costs = covering_cost.covering_cost(gbest)
        out = part_swarm.update_gbest(gbest, gbest_costs, pbest, pbest_costs)
        assert len(out) == 2

    # first element is a ndarray
    def test_update_gbest_first_element_is_ndarray(
            self,
            part_swarm,
            covering_cost,
    ):
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pbest_costs = part_swarm.get_pop_costs(pbest)
        gbest = np.random.randint(0, 2, size=(4, 7))
        gbest_costs = covering_cost.covering_cost(gbest)
        out = part_swarm.update_gbest(gbest, gbest_costs, pbest, pbest_costs)
        assert type(out[0]) == np.ndarray

    # first element is a ndarray of dim (nb_nurses, nb_shifts)
    def test_update_gbest_first_element_is_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            part_swarm,
            covering_cost,
    ):
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pbest_costs = part_swarm.get_pop_costs(pbest)
        gbest = np.random.randint(0, 2, size=(4, 7))
        gbest_costs = covering_cost.covering_cost(gbest)
        out = part_swarm.update_gbest(gbest, gbest_costs, pbest, pbest_costs)
        assert out[0].shape == (4, 7)

    # first element is either 1 or 0
    def test_update_gbest_first_element_is_either_1_or_0(
            self,
            part_swarm,
            covering_cost,
    ):
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pbest_costs = part_swarm.get_pop_costs(pbest)
        gbest = np.random.randint(0, 2, size=(4, 7))
        gbest_costs = covering_cost.covering_cost(gbest)
        out = part_swarm.update_gbest(gbest, gbest_costs, pbest, pbest_costs)
        assert np.all(np.logical_or(out[0] == 1, out[0] == 0))

    # first element is a solution
    def test_update_gbest_first_element_is_a_solution(
            self,
            part_swarm,
            covering_cost,
    ):
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pbest_costs = part_swarm.get_pop_costs(pbest)
        gbest = np.random.randint(0, 2, size=(4, 7))
        gbest_costs = covering_cost.covering_cost(gbest)
        out = part_swarm.update_gbest(gbest, gbest_costs, pbest, pbest_costs)
        assert self.is_solution(out[0])

    # if pbest is a solution with a lower covering cost than gbest, gbest is
    # updated with pbest
    def test_update_gbest_if_pbest_is_a_solution_with_a_lower_covering_cost_than_gbest_gbest_is_updated_with_pbest(
            self,
            part_swarm,
            covering_cost,
    ):
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pbest_costs = part_swarm.get_pop_costs(pbest)
        gbest = np.zeros((4, 7))
        gbest_costs = covering_cost.covering_cost(gbest)
        out = part_swarm.update_gbest(gbest, gbest_costs, pbest, pbest_costs)
        target_out = min(
            pbest,
            key=lambda x: covering_cost.covering_cost(x)
        )
        assert np.array_equal(out[0], target_out)

    # second element is an np.int64
    def test_update_gbest_second_element_is_np_int64(
            self,
            part_swarm,
            covering_cost,
    ):
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pbest_costs = part_swarm.get_pop_costs(pbest)
        gbest = np.random.randint(0, 2, size=(4, 7))
        gbest_costs = covering_cost.covering_cost(gbest)
        out = part_swarm.update_gbest(gbest, gbest_costs, pbest, pbest_costs)
        assert type(out[1]) == np.int64

    # second element is covering cost of first element
    def test_update_gbest_second_element_is_covering_cost_of_first_element(
            self,
            part_swarm,
            covering_cost,
    ):
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pbest_costs = part_swarm.get_pop_costs(pbest)
        gbest = np.random.randint(0, 2, size=(4, 7))
        gbest_costs = covering_cost.covering_cost(gbest)
        out = part_swarm.update_gbest(gbest, gbest_costs, pbest, pbest_costs)
        target_out = covering_cost.covering_cost(out[0])
        assert out[1] == target_out

    # second element is greater than or equal to 0
    def test_update_gbest_second_element_is_greater_than_or_equal_to_0(
            self,
            part_swarm,
            covering_cost,
    ):
        pbest = np.random.randint(0, 2, size=(5, 4, 7))
        pbest_costs = part_swarm.get_pop_costs(pbest)
        gbest = np.random.randint(0, 2, size=(4, 7))
        gbest_costs = covering_cost.covering_cost(gbest)
        out = part_swarm.update_gbest(gbest, gbest_costs, pbest, pbest_costs)
        assert out[1] >= 0

    # update_velocity
    # returns a ndarray
    def test_update_velocity_returns_a_ndarray(self, part_swarm):
        v = np.random.uniform(-1, 1, size=(5, 4, 7))
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        d1, d2 = part_swarm.get_d1_d2(ycompare)
        out = part_swarm.update_velocity(v, d1, d2)
        assert type(out) == np.ndarray

    # returns a ndarray of dim (swarm_size, nb_nurses, nb_shifts)
    def test_update_velocity_returns_a_ndarray_of_dim_swarm_size_nb_nurses_nb_shifts(
            self,
            part_swarm,
    ):
        v = np.random.uniform(-1, 1, size=(5, 4, 7))
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        d1, d2 = part_swarm.get_d1_d2(ycompare)
        out = part_swarm.update_velocity(v, d1, d2)
        assert out.shape == (5, 4, 7)

    # each item of output is between -2.15 and 1.35
    def test_update_velocity_each_item_of_output_is_between_minus_2_15_and_1_35(
            self,
            part_swarm,
    ):
        v = np.random.uniform(-1, 1, size=(5, 4, 7))
        ycompare = np.random.randint(-1, 2, size=(5, 4, 7))
        d1, d2 = part_swarm.get_d1_d2(ycompare)
        out = part_swarm.update_velocity(v, d1, d2)
        assert np.all(out <= 1.35)
        assert np.all(out >= -2.15)

    # particle_swarm_optimization
    # return tuple of len 3
    def test_particle_swarm_optimization_return_tuple_of_len_3(
            self,
            part_swarm,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert len(out) == 3

    # first element is a numpy array
    def test_particle_swarm_optimization_first_element_is_a_numpy_array(
            self,
            part_swarm,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert type(out[0]) == np.ndarray

    # first element is a numpy array of dim (nb_nurses, nb_shifts)
    def test_particle_swarm_optimization_first_element_is_a_numpy_array_of_dim_nb_nurses_nb_shifts(
            self,
            part_swarm,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert out[0].shape == (4, 7)

    # first element is a solution
    def test_particle_swarm_optimization_first_element_is_a_solution(
            self,
            part_swarm,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_particle_swarm_optimization_first_element_comply_with_the_max_work_days_per_week_constraint(
            self,
            part_swarm,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_particle_swarm_optimization_second_element_is_of_type_np_int64(
            self,
            part_swarm,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_particle_swarm_optimization_second_element_is_the_covering_cost_of_the_first_element(
            self,
            part_swarm,
            covering_cost,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert out[1] == covering_cost.covering_cost(out[0])

    # covering cost of first element is less than or equal to the covering cost
    # of the initial solution
    def test_particle_swarm_optimization_covering_cost_of_first_element_is_less_than_or_equal_to_the_covering_cost_of_the_initial_solution(
            self,
            part_swarm,
            covering_cost,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        best_init = min(
            initial_swarm,
            key=lambda x: covering_cost.covering_cost(x),
        )
        best_init_cost = covering_cost.covering_cost(best_init)
        assert covering_cost.covering_cost(out[0]) <= best_init_cost

    # third element is a list
    def test_particle_swarm_optimization_third_element_is_a_list(
            self,
            part_swarm,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_particle_swarm_optimization_third_element_is_a_list_of_np_int64_or_np_inf(
            self,
            part_swarm,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert all(
            type(x) == np.int64 or type(x) == np.inf for x in out[2]
        )

    # third element items are all greater than or equal to 0
    def test_particle_swarm_optimization_third_element_items_are_all_greater_than_or_equal_to_0(
            self,
            part_swarm,
    ):
        initial_swarm = np.zeros((5, 4, 7))
        out = part_swarm.particle_swarm_optimization(initial_swarm)
        assert all(x >= 0 for x in out[2])

    # search_solution
    # return tuple of len 3
    def test_search_solution_return_tuple_of_len_3(self, part_swarm):
        out = part_swarm.search_solution()
        assert len(out) == 3

    # first element is a numpy array
    def test_search_solution_first_element_is_a_numpy_array(self, part_swarm):
        out = part_swarm.search_solution()
        assert type(out[0]) == np.ndarray

    # first element is a numpy array of dim (nb_nurses, nb_shifts)
    def test_search_solution_first_element_is_a_numpy_array_of_dim_nb_nurses_nb_shifts(
            self,
            part_swarm,
    ):
        out = part_swarm.search_solution()
        assert out[0].shape == (4, 7)

    # first element is a solution
    def test_search_solution_first_element_is_a_solution(self, part_swarm):
        out = part_swarm.search_solution()
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_search_solution_first_element_comply_with_the_max_work_days_per_week_constraint(
            self,
            part_swarm,
    ):
        out = part_swarm.search_solution()
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_search_solution_second_element_is_of_type_np_int64(self, part_swarm):
        out = part_swarm.search_solution()
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_search_solution_second_element_is_the_covering_cost_of_the_first_element(
            self,
            part_swarm,
            covering_cost,
    ):
        out = part_swarm.search_solution()
        assert out[1] == covering_cost.covering_cost(out[0])

    # third element is a list
    def test_search_solution_third_element_is_a_list(self, part_swarm):
        out = part_swarm.search_solution()
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_search_solution_third_element_is_a_list_of_np_int64_or_np_inf(
            self,
            part_swarm,
    ):
        out = part_swarm.search_solution()
        assert all(
            type(x) == np.int64 or type(x) == np.inf for x in out[2]
        )

    # third element items are all greater than or equal to 0
    def test_search_solution_third_element_items_are_all_greater_than_or_equal_to_0(
            self,
            part_swarm,
    ):
        out = part_swarm.search_solution()
        assert all(x >= 0 for x in out[2])
