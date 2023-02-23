import matplotlib.pyplot as plt

from genetic_algo.genetic_algo import GeneticAlgo
from tabu_search.tabu_search import TabuSearch
from utils.covering_cost import CoveringCost
from utils.get_population import GetPopulation
from validation import Validation

if __name__ == '__main__':
    # parameters
    # problem parameters
    nb_work_days_per_week = 7      # number of work days per week
    nb_shifts_per_work_day = 1     # number of shifts per work day
    nb_nrs_per_shift = 2           # required number of nurses per shift
    nb_nurses = 4                  # number of nurses
    nrs_max_work_days_per_week = 5  # maximum number of work days per week per nurse

    # tabu search parameters
    nb_iter = 1000                 # number of iterations for tabu search algorithm
    tabu_limit = 10                # max nb of iter solutions stay in memory before forgetting

    # genetic algorithm parameters
    nb_gen = 1000
    pop_size = 100
    nb_parents = 2
    crossover_prob = 0.7
    mutation_prob = 0.3

    # run tabu search
    get_population = GetPopulation(
        nb_nurses,
        nb_work_days_per_week,
        nb_shifts_per_work_day,
        nrs_max_work_days_per_week,
    )

    covering_cost = CoveringCost(
        nb_work_days_per_week,
        nb_shifts_per_work_day,
        nb_nrs_per_shift,
    )


    tabu_search = TabuSearch(
        nb_nurses,
        nb_work_days_per_week,
        nb_shifts_per_work_day,
        nb_nrs_per_shift,
        nrs_max_work_days_per_week,
        nb_iter,
        tabu_limit,
        get_population,
        covering_cost,
    )

    genetic_algorithm = GeneticAlgo(
        nb_nurses,
        nb_work_days_per_week,
        nb_shifts_per_work_day,
        nb_nrs_per_shift,
        nrs_max_work_days_per_week,
        nb_gen,
        pop_size,
        nb_parents,
        crossover_prob,
        mutation_prob,
        get_population,
        covering_cost,
    )

    # solution, solution_cost, states = tabu_search.search_solution()
    solution, solution_cost, states = genetic_algorithm.search_solution()

    # validate solution
    validation = Validation(
        nb_nurses,
        nb_work_days_per_week,
        nb_shifts_per_work_day,
        nb_nrs_per_shift,
        nrs_max_work_days_per_week,
    )

    validation_object, overall_validation = validation.validate_solution(
        solution,
        solution_cost,
        states,
    )

    # print results
    print('solution:', solution)
    print('solution_cost:', solution_cost)
    print('solution has been validated:', overall_validation)
    print('validation object:', validation_object)

    plt.plot(states)
    plt.xlabel('nb iterations')
    plt.ylabel('coverage cost')
    plt.show()
