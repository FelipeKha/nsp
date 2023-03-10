import argparse
import matplotlib.pyplot as plt
import subprocess
import time

from google_cp_sat.cp_sat import CPSAT
from genetic_algo.genetic_algo import GeneticAlgo
from particle_swarm.pso import ParticleSwarmOptimization
from problem_setup.problem import Problem
# from reinforcement_learning.rf_rnn import ReingforcementLearningRNN
from simulated_annealing.sim_anneal import SimulatedAnnealing
from tabu_search.tabu_search import TabuSearch
from utils.benchmark_track import benchmark_track
from utils.check_constraints import check_population_for_max_days_per_week
from utils.covering_cost import covering_cost
from utils.iter_next import IterNext
from utils.get_neighbour import get_neighbour, get_neighbour_tabu
from utils.get_population import \
    get_random_initial_solution, \
    get_initial_population
from validation import Validation


validation = Validation()
iter_next = IterNext(
    nb_iter_max=1000,
    zero_cost=True,
    validation=True,
    zero_cost_max=0,
    mean_hist_cost=[0, 1.0],
    check_every=1,
)


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
        'cp_sat',
    ],
    help='algorithm to use',
)

args = parser.parse_args()

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
        validation=validation,
        iter_next=iter_next,
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
    'simulated_annealing': SimulatedAnnealing(
        nb_iter=1000,
        nb_neighbours=10,
        k=20,
        lam=0.005,
        limit=100,
        get_random_initial_solution=get_random_initial_solution,
        get_neighbour=get_neighbour,
        covering_cost=covering_cost,
    ),
    'particle_swarm_optimization': ParticleSwarmOptimization(
        swarm_size=20,
        max_iter=1000,
        c1=0.7,
        c2=0.3,
        w=0.75,
        alpha=0.3,
        get_initial_population=get_initial_population,
        covering_cost=covering_cost,
        check_population_for_max_days_per_week=check_population_for_max_days_per_week,
    ),
    'cp_sat': CPSAT(),
    # 'reinforcement_learning': ReingforcementLearningRNN,
}



def search_solution(problem_in: str, algo_in: str):
    problem = problems[problem_in]
    algo = algos[algo_in]

    start_time = time.time()
    solution, solution_cost, states, nb_iter, validate, validation_details = algo(problem)
    end_time = time.time()
    search_time = end_time - start_time

    benchmark_track(
        problem,
        algo,
        solution_cost,
        search_time,
        nb_iter,
        validate,
        validation_details,
    )
    
    
    # validation, validation_details = validation(
    #     solution,
    #     solution_cost,
    #     states,
    #     problem,
    # )

    print('solution:')
    print(solution)
    print('solution_cost:', solution_cost)
    print('search_time (s):', search_time)
    print('nb_iter:', nb_iter)
    print('validation:', validation)
    print('validation_details:')
    print(validation_details)

search_solution(args.problem, args.algo)

#     plt.plot(states)
#     plt.xlabel('nb iterations')
#     plt.ylabel('coverage cost')
#     plt.show()
