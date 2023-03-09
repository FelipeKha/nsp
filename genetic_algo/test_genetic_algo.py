import numpy as np
import random
import pytest

from genetic_algo.genetic_algo import GeneticAlgo
from problem_setup.problem import Problem
from utils.covering_cost import covering_cost
from utils.get_population import get_initial_population


class TestGeneticAlgo:
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
    def genetic_algo(self):
        return GeneticAlgo(
            nb_gen=20,
            pop_size=10,
            nb_parents_mating=2,
            crossover_prob=0.7,
            mutation_prob=0.3,
            get_initial_population=get_initial_population,
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

    # select_parent
    # return a numpy array
    def test_select_parent(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.select_parent(population, problem)
        assert type(out) == np.ndarray

    # return a numpy array with shape (nb_parents_mating, nb_nurses, nb_work_days_per_week * nb_shifts_per_work_day)
    def test_select_parent_shape(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.select_parent(population, problem)
        assert out.shape == (2, 4, 7)

    # each item of the returned array is a solution
    def test_select_parent_is_solution(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.select_parent(population, problem)
        for solution in out:
            assert self.is_solution(solution)

    # each item of the returned array is in the input population
    def test_select_parent_is_in_population(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.select_parent(population, problem)
        for solution in out:
            assert np.any(np.all(population == solution, axis=(1, 2)))

    # crossover
    # return a numpy array
    def test_crossover(self, genetic_algo, problem):
        parents = np.zeros((2, 4, 7))
        out = genetic_algo.crossover(parents, problem)
        assert type(out) == np.ndarray

    # return a numpy array with shape (nb_parents_mating, nb_nurses, nb_work_days_per_week * nb_shifts_per_work_day)
    def test_crossover_shape(self, genetic_algo, problem):
        parents = np.zeros((2, 4, 7))
        out = genetic_algo.crossover(parents, problem)
        assert out.shape == (2, 4, 7)

    # each item of the returned array is a solution
    def test_crossover_is_solution(self, genetic_algo, problem):
        parents = np.zeros((2, 4, 7))
        out = genetic_algo.crossover(parents, problem)
        for solution in out:
            assert self.is_solution(solution)

    # children number of nurse working on any shift is less than or equal to nb_nrs_per_shift, or their are equal to the parents
    def test_crossover_nb_nrs_per_shift(self, genetic_algo, problem):
        parents = np.zeros((2, 4, 7))
        out = genetic_algo.crossover(parents, problem)
        if np.amax(out.sum(axis=1)) > 2:
            assert np.array_equal(out, parents)
        else:
            assert np.amax(out.sum(axis=1)) <= 2

    # mutation
    # return a numpy array
    def test_mutation(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.mutation(population, problem)
        assert type(out) == np.ndarray

    # return a numpy array with shape (pop_size, nb_nurses, nb_work_days_per_week * nb_shifts_per_work_day)
    def test_mutation_shape(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.mutation(population, problem)
        assert out.shape == (10, 4, 7)

    # each item of the returned array is a solution
    def test_mutation_is_solution(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.mutation(population, problem)
        for solution in out:
            assert self.is_solution(solution)

    # each item of the returned array comply with the max work days per week constraint
    def test_mutation_max_work_days_per_week(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.mutation(population, problem)
        assert out.sum(axis=1).max() <= 5

    # get_index_lowest_cost_pop
    # return an np.int64
    def test_get_index_lowest_cost_pop_return_np_int64(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.get_index_lowest_cost_pop(population, problem)
        assert type(out) == np.int64

    # return an int between 0 and pop_size - 1
    def test_get_index_lowest_cost_pop_range(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.get_index_lowest_cost_pop(population, problem)
        assert 0 <= out <= 9

    # return the index of the solution with the lowest covering cost in the input population
    def test_get_index_lowest_cost_pop_index(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        target_index = random.randint(0, 9)
        population[target_index] = [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
        ]
        out = genetic_algo.get_index_lowest_cost_pop(population, problem)
        assert out == target_index

    # trim_population
    # return a numpy array
    def test_trim_population(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.trim_population(population, problem)
        assert type(out) == np.ndarray

    # return a numpy array with shape (pop_size, nb_nurses, nb_work_days_per_week * nb_shifts_per_work_day)
    def test_trim_population_shape(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.trim_population(population, problem)
        assert out.shape == (10, 4, 7)

    # each item of the returned array is a solution
    def test_trim_population_is_solution(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.trim_population(population, problem)
        for solution in out:
            assert self.is_solution(solution)

    # if the input population has a size greater than pop_size, the returned array has a size of pop_size
    def test_trim_population_size(self, genetic_algo, problem):
        population = np.zeros((12, 4, 7))
        out = genetic_algo.trim_population(population, problem)
        assert out.shape[0] == 10

    # if the input population has a size greater than pop_size, the removed individuals are the ones with the highest covering cost
    def test_trim_population_remove_high_cost(self, genetic_algo, problem):
        zero_cost_sol = [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
        ]
        zero_sol = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        population = [zero_cost_sol] * 10 + [zero_sol] * 2
        population = np.array(population)
        out = genetic_algo.trim_population(population, problem)
        target_out = [zero_cost_sol] * 10
        target_out = np.array(target_out)
        assert np.array_equal(out, target_out)
    # if the input population has a size greater than pop_size, the returned array is contained in the input population
    def test_trim_population_contained(self, genetic_algo, problem):
        zero_cost_sol = [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
        ]
        zero_sol = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        population = [zero_cost_sol] * 10 + [zero_sol] * 2
        population = np.array(population)
        out = genetic_algo.trim_population(population, problem)
        result = np.isin(population, out)
        assert np.all(result)


    # simple_genetic_algorithm
    # return tuple of len 3
    def test_simple_genetic_algorithm(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        assert type(out) == tuple
        assert len(out) == 3

    # first element is a numpy array
    def test_simple_genetic_algorithm_first_element(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        assert type(out[0]) == np.ndarray

    # first element is a solution
    def test_simple_genetic_algorithm_first_element_is_solution(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_simple_genetic_algorithm_first_element_max_work_days_per_week(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_simple_genetic_algorithm_second_element(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_simple_genetic_algorithm_second_element_covering_cost(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        assert out[1] == covering_cost(out[0], problem)

    # covering cost of first element is less than or equal to the covering cost
    # of the initial solution lowest covering cost
    def test_simple_genetic_algorithm_second_element_covering_cost_less_than_lowest_covering_cost(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        best_solution = min(population, key=lambda x : covering_cost(x, problem))
        best_coverage_cost = covering_cost(best_solution, problem)
        assert out[1] <= best_coverage_cost

    # third element is a list
    def test_simple_genetic_algorithm_third_element(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_simple_genetic_algorithm_third_element_list(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        assert all(type(x) == np.int64 or type(x) == np.inf for x in out[2])

    # third element items are all greater than or equal to 0
    def test_simple_genetic_algorithm_third_element_list(self, genetic_algo, problem):
        population = np.zeros((10, 4, 7))
        out = genetic_algo.simple_genetic_algorithm(population, problem)
        assert all(x >= 0 for x in out[2])

    # __call__
    # return tuple of len 3
    def test_search_solution(self, genetic_algo, problem):
        out = genetic_algo(problem)
        assert type(out) == tuple
        assert len(out) == 3

    # first element is a numpy array
    def test_search_solution_first_element(self, genetic_algo, problem):
        out = genetic_algo(problem)
        assert type(out[0]) == np.ndarray

    # first element is a solution
    def test_search_solution_first_element_is_solution(self, genetic_algo, problem):
        out = genetic_algo(problem)
        assert self.is_solution(out[0])

    # first element comply with the max work days per week constraint
    def test_search_solution_first_element_max_work_days_per_week(self, genetic_algo, problem):
        out = genetic_algo(problem)
        assert out[0].sum(axis=1).max() <= 5

    # second element is of type np.int64
    def test_search_solution_second_element(self, genetic_algo, problem):
        out = genetic_algo(problem)
        assert type(out[1]) == np.int64

    # second element is the covering cost of the first element
    def test_search_solution_second_element_covering_cost(self, genetic_algo, problem):
        out = genetic_algo(problem)
        assert out[1] == covering_cost(out[0], problem)

    # third element is a list
    def test_search_solution_third_element(self, genetic_algo, problem):
        out = genetic_algo(problem)
        assert type(out[2]) == list

    # third element is a list of np.int64 or np.inf
    def test_search_solution_third_element_list(self, genetic_algo, problem):
        out = genetic_algo(problem)
        assert all(type(x) == np.int64 or type(x) == np.inf for x in out[2])

    # third element items are all greater than or equal to 0
    def test_search_solution_third_element_list(self, genetic_algo, problem):
        out = genetic_algo(problem)
        assert all(x >= 0 for x in out[2])

    ###########
    # simulated_annealing
    # return tuple of len 3

    # first element is a numpy array

    # first element is a solution

    # first element comply with the max work days per week constraint

    # second element is of type np.int64

    # second element is the covering cost of the first element

    # covering cost of first element is less than or equal to the covering cost
    # of the initial solution

    # third element is a list

    # third element is a list of np.int64 or np.inf

    # third element items are all greater than or equal to 0

    # search_solution
    # return tuple of len 3

    # first element is a numpy array

    # first element is a solution

    # first element comply with the max work days per week constraint

    # second element is of type np.int64

    # second element is the covering cost of the first element

    # third element is a list

    # third element is a list of np.int64 or np.inf

    # third element items are all greater than or equal to 0
