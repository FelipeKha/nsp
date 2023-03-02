import matplotlib.pyplot as plt

from genetic_algo.genetic_algo import GeneticAlgo

from simulated_annealing.sim_anneal import SimulatedAnnealing
from tabu_search.tabu_search import TabuSearch
from utils.covering_cost import CoveringCost
from utils.get_neighbour import GetNeighbour
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

    # simulated annealing parameters
    nb_neighbours = 10
    # annealing schedule parameters (function that determine the temperature, 
    # the lower it is the less likely we are to accept a worse solution)
    # temperature = k * exp(-lam * iter_num) if iter_num < limit, else 0
    k = 20      # scale up the temperature
    lam = 0.005 # speed of decay of temperature (the lower the slower)
    limit = 100 # number of iterations after which no non_improving moves are accepted

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


    get_neighbour = GetNeighbour(
        nb_nurses,
        nb_work_days_per_week,
        nb_shifts_per_work_day,
        nrs_max_work_days_per_week,
        covering_cost,
    )

    tabu_search = TabuSearch(
        nb_nurses,
        nb_work_days_per_week,
        nb_shifts_per_work_day,
        nb_nrs_per_shift,
        nrs_max_work_days_per_week,
        nb_iter,
        nb_neighbours,
        tabu_limit,
        get_population,
        get_neighbour,
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

    simulated_annealing = SimulatedAnnealing(
        nb_nurses,
        nb_work_days_per_week,
        nb_shifts_per_work_day,
        nb_nrs_per_shift,
        nrs_max_work_days_per_week,
        nb_iter,
        nb_neighbours,
        k, 
        lam, 
        limit,
        get_population,
        get_neighbour,
        covering_cost,
    )

    # solution, solution_cost, states = tabu_search.search_solution()
    # solution, solution_cost, states = genetic_algorithm.search_solution()
    solution, solution_cost, states = simulated_annealing.search_solution()

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
