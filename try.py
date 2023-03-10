from problem_setup.problem import Problem
from tabu_search.tabu_search import TabuSearch
from utils.benchmark_track import benchmark_track, get_algo_params
from utils.covering_cost import covering_cost
from utils.get_neighbour import get_neighbour_tabu
from utils.get_population import get_random_initial_solution
from utils.iter_next import IterNext
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

problem = Problem(
        nb_nurses=4,
        nb_work_days_per_week=7,
        nb_shifts_per_work_day=1,
        target_nb_nrs_per_shift=2,
        nrs_max_work_days_per_week=5,
    )

algo = TabuSearch(
        nb_iter=10,
        nb_neighbours=10,
        tabu_limit=10,
        get_random_initial_solution=get_random_initial_solution,
        get_neighbour_tabu=get_neighbour_tabu,
        covering_cost=covering_cost,
        validation=validation,
        iter_next=iter_next,
    )

# benchmark_track(
#     problem=problem,
#     algo=algo,
#     solution_cost=0,
#     search_time=18,
#     nb_iter=9,
#     validation=True,
#     validation_details={'test1': True, 'test2': False},
# )

get_algo_params(algo)