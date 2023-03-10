import numpy as np
import random

from problem_setup.problem import Problem

class GeneticAlgo:
    def __init__(
        self,
        nb_gen: int,
        pop_size: int,
        nb_parents_mating: int,
        crossover_prob: float,
        mutation_prob: float,
        get_initial_population: callable,
        covering_cost: callable,
    ) -> None:
        self.nb_gen = nb_gen
        self.pop_size = pop_size
        self.nb_parents_mating = nb_parents_mating
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.get_initial_population = get_initial_population
        self.covering_cost = covering_cost

    def __call__(
        self,
        problem: Problem,
    ) -> tuple[np.ndarray, int, list]:
        population = self.get_initial_population(self.pop_size, problem)
        best_solution, best_solution_cost, states = self.simple_genetic_algorithm(
            population, problem)
        return best_solution, best_solution_cost, states

    def select_parent(
            self,
            population: np.ndarray,
            problem: Problem,
    ) -> np.ndarray:
        """
        Given a population of dim (pop_size, nb_nurses, nb_shifts), returns 
        np array parents of dim (num_parents_mating, nb_nurses, nb_shifts)
        """
        parents = np.empty(
            (
                self.nb_parents_mating,
                problem.nb_nurses,
                problem.nb_work_days_per_week * problem.nb_shifts_per_work_day
            ),
            dtype=int
        )
        parent_indexes = np.random.randint(
            0,
            self.pop_size,
            self.nb_parents_mating,
        )
        for index, element in np.ndenumerate(parent_indexes):
            parents[index] = population[element]
        return parents

    def crossover(
            self,
            parents: np.ndarray,
            problem: Problem,
    ) -> np.ndarray:
        """
        Given parents of dim (num_parents_mating, nb_nurses, nb_shifts), returns
        children of dim (num_parents_mating, nb_nurses, nb_shifts) with 
        crossover rate of crossover_prob.
        The crossover is done by selecting a random nurse within parents, and 
        crossing its schedules between parents.
        """
        chromosome_length = problem.nb_work_days_per_week * \
            problem.nb_shifts_per_work_day
        cross_points = np.random.randint(
            0,
            chromosome_length,
            self.nb_parents_mating - 1,
        )
        cross_points = np.sort(cross_points)
        if self.crossover_prob > random.random():
            nrs = np.random.randint(0, problem.nb_nurses - 1)
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
            if np.amax(children.sum(axis=1)) > problem.target_nb_nrs_per_shift:
                return parents
            else:
                return children
        else:
            return parents

    def mutation(
            self,
            population: np.ndarray,
            problem: Problem,
    ) -> np.ndarray:
        """
        Given a population of dim (pop_size, nb_nurses, nb_shifts), returns a 
        population of dim (pop_size, nb_nurses, nb_shifts) with mutation rate of
        mutation_prob
        """
        chromosome_length = problem.nb_work_days_per_week * \
            problem.nb_shifts_per_work_day
        for i in range(population.shape[0] - 1):
            for j in range(problem.nb_nurses - 1):
                if self.mutation_prob > random.random():
                    index = np.random.randint(0, chromosome_length)
                    temp = population.copy()[i][j]
                    temp[index] = 1 - temp[index]
                    if temp.sum(axis=0) <= problem.nrs_max_work_days_per_week:
                        population[i][j] = temp
        return population

    def get_index_lowest_cost_pop(
            self,
            population: np.ndarray,
            problem: Problem,
    ) -> int:
        """
        Given a population of dim (pop_size, nb_nurses, nb_shifts), returns the 
        index of the individual with the lowest covering cost
        """
        covering_costs = np.array(
            [self.covering_cost(
                solution, problem) for solution in population]
        )
        index_lowest_cost_pop = np.argmin(covering_costs)
        return index_lowest_cost_pop

    def trim_population(
            self,
            population: np.ndarray,
            problem: Problem,
    ) -> np.ndarray:
        """
        Given a population of dim (more than pop_size, nb_nurses, nb_shifts), returns a 
        population of dim (pop_size, nb_nurses, nb_shifts) with the lowest 
        covering cost individuals
        """
        covering_costs = np.array(
            [self.covering_cost(
                solution, problem) for solution in population]
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
        problem: Problem,
    ) -> tuple[np.ndarray, int, list]:
        states = []
        best_solution = np.empty(
            (
                problem.nb_nurses,
                problem.nb_work_days_per_week * problem.nb_shifts_per_work_day
            ),
            dtype=int
        )
        best_solution_cost = np.inf

        for i in range(self.nb_gen):
            print(i)
            parents = self.select_parent(population, problem)
            children = self.crossover(parents, problem)
            population = np.concatenate((population, children))
            population = self.mutation(population, problem)
            current_best_solution_index = self.get_index_lowest_cost_pop(
                population,
                problem,
            )
            current_best_solution = population[current_best_solution_index]
            current_best_solution_cost = self.covering_cost(
                current_best_solution,
                problem,
            )
            if current_best_solution_cost < best_solution_cost:
                best_solution = current_best_solution
                best_solution_cost = current_best_solution_cost
            states.append(current_best_solution_cost)
            population = self.trim_population(population, problem)

        return best_solution, best_solution_cost, states