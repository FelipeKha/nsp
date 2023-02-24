import numpy as np
import random

from utils.covering_cost import CoveringCost
from utils.get_population import GetPopulation


class GeneticAlgo:
    def __init__(
        self,
        nb_nurses: int,
        nb_work_days_per_week: int,
        nb_shifts_per_work_day: int,
        nb_nrs_per_shift: int,
        nrs_max_work_days_per_week: int,
        nb_gen: int,
        pop_size: int,
        num_parents_mating: int,
        crossover_prob: float,
        mutation_prob: float,
        get_population: GetPopulation,
        covering_cost: CoveringCost,
    ) -> None:
        self.nb_nurses = nb_nurses
        self.nb_work_days_per_week = nb_work_days_per_week
        self.nb_shifts_per_work_day = nb_shifts_per_work_day
        self.nb_nrs_per_shift = nb_nrs_per_shift
        self.nrs_max_work_days_per_week = nrs_max_work_days_per_week
        self.nb_gen = nb_gen
        self.pop_size = pop_size
        self.num_parents_mating = num_parents_mating
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.get_population = get_population
        self.covering_cost = covering_cost

    def select_parent(self, population: np.ndarray) -> np.ndarray:
        """
        Given a population of dim (pop_size, nb_nurses, nb_shifts), returns 
        np array parents of dim (num_parents_mating, nb_nurses, nb_shifts)
        """
        parents = np.empty(
            (
                self.num_parents_mating,
                self.nb_nurses,
                self.nb_work_days_per_week * self.nb_shifts_per_work_day
            ),
            dtype=int
        )
        parent_indexes = np.random.randint(
            0,
            self.pop_size,
            self.num_parents_mating,
        )
        for index, element in np.ndenumerate(parent_indexes):
            parents[index] = population[element]
        return parents

    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        Given parents of dim (num_parents_mating, nb_nurses, nb_shifts), returns
        children of dim (num_parents_mating, nb_nurses, nb_shifts) with 
        crossover rate of crossover_prob.
        The crossover is done by selecting a random nurse within parents, and 
        crossing its schedules between parents.
        """
        chromosome_length = self.nb_work_days_per_week * self.nb_shifts_per_work_day
        cross_points = np.random.randint(
            0,
            chromosome_length,
            self.num_parents_mating - 1,
        )
        cross_points = np.sort(cross_points)
        if self.crossover_prob > random.random():
            nrs = np.random.randint(0, self.nb_nurses - 1)
            child1 = parents[0]
            child2 = parents[1]
            child1[nrs] = np.concatenate((
                parents[0][nrs][0: cross_points[0]],
                parents[1][nrs][cross_points[0]: chromosome_length],
            ))
            child2[nrs] = np.concatenate((
                parents[1][nrs][0: cross_points[0]],
                parents[0][nrs][cross_points[0]: chromosome_length],
            ))
            children = np.array([child1, child2])
            if np.amax(children.sum(axis=1)) > self.nb_nrs_per_shift:
                return parents
            else:
                return children
        else:
            return parents

    def mutation(self, population: np.ndarray) -> np.ndarray:
        """
        Given a population of dim (pop_size, nb_nurses, nb_shifts), returns a 
        population of dim (pop_size, nb_nurses, nb_shifts) with mutation rate of
        mutation_prob
        """
        chromosome_length = self.nb_work_days_per_week * self.nb_shifts_per_work_day
        for i in range(population.shape[0] - 1):
            for j in range(self.nb_nurses - 1):
                if self.mutation_prob > random.random():
                    index = np.random.randint(0, chromosome_length)
                    temp = population.copy()[i][j]
                    temp[index] = 1 - temp[index]
                    if temp.sum(axis=0) <= self.nrs_max_work_days_per_week:
                        population[i][j] = temp
        return population

    def get_index_lowest_cost_pop(self, population: np.ndarray) -> int:
        """
        Given a population of dim (pop_size, nb_nurses, nb_shifts), returns the 
        index of the individual with the lowest covering cost
        """
        covering_costs = np.array(
            [self.covering_cost.covering_cost(
                solution) for solution in population]
        )
        index_lowest_cost_pop = np.argmin(covering_costs)
        return index_lowest_cost_pop

    def trim_population(self, population: np.ndarray) -> np.ndarray:
        """
        Given a population of dim (more than pop_size, nb_nurses, nb_shifts), returns a 
        population of dim (pop_size, nb_nurses, nb_shifts) with the lowest 
        covering cost individuals
        """
        covering_costs = np.array(
            [self.covering_cost.covering_cost(
                solution) for solution in population]
        )
        while population.shape[0] > self.pop_size:
            index_highest_cost_pop = np.argmax(covering_costs)
            population = np.delete(population, index_highest_cost_pop, axis=0)
            covering_costs = np.delete(
                covering_costs, index_highest_cost_pop, axis=0)
        return population

    def simple_genetic_algorithm(
        self,
        population: np.ndarray,
    ) -> tuple[np.ndarray, int, list]:
        states = []
        best_solution = np.empty(
            (
                self.nb_nurses,
                self.nb_work_days_per_week * self.nb_shifts_per_work_day
            ),
            dtype=int
        )
        best_solution_cost = np.inf

        for i in range(self.nb_gen):
            print(i)
            parents = self.select_parent(population)
            children = self.crossover(parents)
            population = np.concatenate((population, children))
            population = self.mutation(population)
            current_best_solution_index = self.get_index_lowest_cost_pop(
                population)
            current_best_solution = population[current_best_solution_index]
            current_best_solution_cost = self.covering_cost.covering_cost(
                current_best_solution)
            if current_best_solution_cost < best_solution_cost:
                best_solution = current_best_solution
                best_solution_cost = current_best_solution_cost
            states.append(current_best_solution_cost)
            population = self.trim_population(population)

        return best_solution, best_solution_cost, states

    def search_solution(self) -> tuple[np.ndarray, int, list]:
        population = self.get_population.get_initial_population(self.pop_size)
        best_solution, best_solution_cost, states = self.simple_genetic_algorithm(
            population)
        return best_solution, best_solution_cost, states
