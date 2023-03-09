import argparse
import matplotlib.pyplot as plt

# from google_cp_sat.cp_sat import CPSAT
from genetic_algo.genetic_algo import GeneticAlgo
# from particle_swarm.pso import ParticleSwarmOptimization
from problem_setup.problem import Problem
# from reinforcement_learning.rf_rnn import ReingforcementLearningRNN
# from simulated_annealing.sim_anneal import SimulatedAnnealing
from tabu_search.tabu_search import TabuSearch
# from utils.check_constraints import CheckConstraints
from utils.covering_cost import covering_cost
from utils.get_neighbour import get_neighbour_tabu
from utils.get_population import \
    get_random_initial_solution, \
        get_initial_population
# from validation import Validation

parser = argparse.ArgumentParser(
    prog='nurse_rostering',
    description='Solve nurse rostering problem with different algorithms',
    epilog='Enjoy the program!',
)

parser.add_argument(
    '--problem',
    type=str,
    default='small',
    choices=['small', 'medium', 'large'],
    help='problem to solve',
)

parser.add_argument(
    '--algo',
    type=str,
    default='tabu_search',
    choices=[
        'tabu_search',
        'genetic',
        'simulated_annealing',
        'particle_swarm_optimization',
        'reinforcement_learning',
    ],
    help='algorithm to use',
)

args = parser.parse_args()
print('args:', args)
print('args.problem:', args.problem)
print('args.algo:', args.algo)

problems = {
    'small': Problem(
        nb_nurses=4,
        nb_work_days_per_week=7,
        nb_shifts_per_work_day=1,
        target_nb_nrs_per_shift=2,
        nrs_max_work_days_per_week=5,
    ),
    'medium': Problem(
        nb_nurses=4,
        nb_work_days_per_week=7,
        nb_shifts_per_work_day=1,
        target_nb_nrs_per_shift=2,
        nrs_max_work_days_per_week=5,
    ),
    'large': Problem(
        nb_nurses=4,
        nb_work_days_per_week=7,
        nb_shifts_per_work_day=1,
        target_nb_nrs_per_shift=2,
        nrs_max_work_days_per_week=5,
    ),
}

algos = {
    'tabu_search': TabuSearch(
        nb_iter=1000,
        nb_neighbours=10,
        tabu_limit=10,
        get_random_initial_solution=get_random_initial_solution,
        get_neighbour_tabu=get_neighbour_tabu,
        covering_cost=covering_cost,
    ),
    'genetic': GeneticAlgo(
        nb_gen=1000,
        pop_size=100,
        nb_parents_mating=2,
        crossover_prob=0.7,
        mutation_prob=0.3,
        get_initial_population=get_initial_population,
        covering_cost=covering_cost,
    ),
    # 'simulated_annealing': SimulatedAnnealing,
    # 'particle_swarm_optimization': ParticleSwarmOptimization,
    # 'cpsat': CPSAT,
    # 'reinforcement_learning': ReingforcementLearningRNN,
}

# input from Adrien
# problems = {
#     'small': Problem(1, 3, 3),
#     'medium': Problem(3, 4, 5, otherParama),
# }

# algos = {
#     'genetic': genetic_alg,
# }

# algos[argv[2]](problems[argv[1]])

print(problems[args.problem]())

solution, solution_cost, states = algos[args.algo](problems[args.problem])

print('solution:')
print(solution)
print('solution_cost:', solution_cost)



# if __name__ == '__main__':
#     params_small = {
#         'params_problem': {
#             'nb_nurses': 4,
#             'nb_work_days_per_week': 7,
#             'nb_shifts_per_work_day': 1,
#             'target_nb_nrs_per_shift': 2,
#             'nrs_max_work_days_per_week': 5,
#         },
#         'params_algo': {
#             'tabu_search': {
#                 'nb_iter': 1000,
#                 'tabu_limit': 10,
#             },
#             'genetic_algo': {
#                 'nb_gen': 1000,
#                 'pop_size': 100,
#                 'nb_parents': 2,
#                 'crossover_prob': 0.7,
#                 'mutation_prob': 0.3,
#             },
#             'simulated_annealing': {
#                 'nb_iter': 1000,
#                 'nb_neighbours': 10,
#                 'k': 20,
#                 'lam': 0.005,
#                 'limit': 100,
#             },
#             'particle_swarm_optimization': {
#                 'swarm_size': 20,
#                 'max_iter': 1000,
#                 'c1': 0.7,
#                 'c2': 0.3,
#                 'w': 0.75,
#                 'alpha': 0.3,
#             },
#             'reinforcement_learning': {
#                 'H': 200,
#                 'batch_size': 10,
#                 'learning_rate': 1e-4,
#                 'gamma': 0.99,
#                 'decay_rate': 0.99,
#                 'resume': False,
#             },
#         }
#     }


#     # parameters
#     # problem parameters
#     nb_work_days_per_week = 7      # number of work days per week
#     nb_shifts_per_work_day = 1     # number of shifts per work day
#     nb_nrs_per_shift = 2           # required number of nurses per shift
#     nb_nurses = 4                  # number of nurses
#     nrs_max_work_days_per_week = 5  # maximum number of work days per week per nurse

#     # tabu search parameters
#     nb_iter = 1000                 # number of iterations for tabu search algorithm
#     tabu_limit = 10                # max nb of iter solutions stay in memory before forgetting

#     # genetic algorithm parameters
#     nb_gen = 1000
#     pop_size = 100
#     nb_parents = 2
#     crossover_prob = 0.7
#     mutation_prob = 0.3

#     # simulated annealing parameters
#     nb_neighbours = 10
#     # annealing schedule parameters (function that determine the temperature,
#     # the lower it is the less likely we are to accept a worse solution)
#     # temperature = k * exp(-lam * iter_num) if iter_num < limit, else 0
#     k = 20      # scale up the temperature
#     lam = 0.005 # speed of decay of temperature (the lower the slower)
#     limit = 100 # number of iterations after which no non_improving moves are accepted

#     # particle swarm optimization parameters
#     swarm_size = 20
#     max_iter = 1000
#     c1 = 0.7
#     c2 = 0.3
#     w = 0.75
#     alpha = 0.3

#     # Reinforcement learning hyperparameters
#     H = 200
#     batch_size = 10
#     learning_rate = 1e-4
#     gamma = 0.99
#     decay_rate = 0.99
#     resume = False

#     # run tabu search
#     get_population = GetPopulation(
#         nb_nurses,
#         nb_work_days_per_week,
#         nb_shifts_per_work_day,
#         nrs_max_work_days_per_week,
#     )

#     covering_cost = CoveringCost(
#         nb_work_days_per_week,
#         nb_shifts_per_work_day,
#         nb_nrs_per_shift,
#     )

#     get_neighbour = GetNeighbour(
#         nb_nurses,
#         nb_work_days_per_week,
#         nb_shifts_per_work_day,
#         nrs_max_work_days_per_week,
#         covering_cost,
#     )

#     check_constraints = CheckConstraints()

#     tabu_search = TabuSearch(
#         nb_nurses,
#         nb_work_days_per_week,
#         nb_shifts_per_work_day,
#         nb_nrs_per_shift,
#         nrs_max_work_days_per_week,
#         nb_iter,
#         nb_neighbours,
#         tabu_limit,
#         get_population,
#         get_neighbour,
#         covering_cost,
#     )

#     genetic_algorithm = GeneticAlgo(
#         nb_nurses,
#         nb_work_days_per_week,
#         nb_shifts_per_work_day,
#         nb_nrs_per_shift,
#         nrs_max_work_days_per_week,
#         nb_gen,
#         pop_size,
#         nb_parents,
#         crossover_prob,
#         mutation_prob,
#         get_population,
#         covering_cost,
#     )

#     simulated_annealing = SimulatedAnnealing(
#         nb_nurses,
#         nb_work_days_per_week,
#         nb_shifts_per_work_day,
#         nb_nrs_per_shift,
#         nrs_max_work_days_per_week,
#         nb_iter,
#         nb_neighbours,
#         k,
#         lam,
#         limit,
#         get_population,
#         get_neighbour,
#         covering_cost,
#     )

#     particle_swarm_optimization = ParticleSwarmOptimization(
#         nb_nurses,
#         nb_work_days_per_week,
#         nb_shifts_per_work_day,
#         nb_nrs_per_shift,
#         nrs_max_work_days_per_week,
#         swarm_size,
#         max_iter,
#         c1,
#         c2,
#         w,
#         alpha,
#         get_population,
#         covering_cost,
#         check_constraints,
#     )

#     cpsat = CPSAT(
#         nb_nurses,
#         nb_work_days_per_week,
#         nb_shifts_per_work_day,
#         nb_nrs_per_shift,
#         nrs_max_work_days_per_week,
#     )

#     # solution, solution_cost, states = tabu_search.search_solution()
#     # solution, solution_cost, states = genetic_algorithm.search_solution()
#     # solution, solution_cost, states = simulated_annealing.search_solution()
#     solution, solution_cost, states = particle_swarm_optimization.search_solution()
#     # solution, solution_cost, states = cpsat.search_solution()

#     # validate solution
#     validation = Validation(
#         nb_nurses,
#         nb_work_days_per_week,
#         nb_shifts_per_work_day,
#         nb_nrs_per_shift,
#         nrs_max_work_days_per_week,
#     )

#     validation_object, overall_validation = validation.validate_solution(
#         solution,
#         solution_cost,
#         states,
#     )

#     # print results
#     print('solution:', solution)
#     print('solution_cost:', solution_cost)
#     print('solution has been validated:', overall_validation)
#     print('validation object:', validation_object)

#     plt.plot(states)
#     plt.xlabel('nb iterations')
#     plt.ylabel('coverage cost')
#     plt.show()
