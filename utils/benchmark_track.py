import csv
import datetime
import subprocess

from problem_setup.problem import Problem
from tabu_search.tabu_search import TabuSearch
from utils.covering_cost import covering_cost
from utils.get_neighbour import get_neighbour_tabu
from utils.get_population import get_random_initial_solution


# problem = Problem(
#         nb_nurses=4,
#         nb_work_days_per_week=7,
#         nb_shifts_per_work_day=1,
#         target_nb_nrs_per_shift=2,
#         nrs_max_work_days_per_week=5,
#     )

# algo = TabuSearch(
#         nb_iter=10,
#         nb_neighbours=10,
#         tabu_limit=10,
#         get_random_initial_solution=get_random_initial_solution,
#         get_neighbour_tabu=get_neighbour_tabu,
#         covering_cost=covering_cost,
#     )


def benchmark_track(
        problem,
        algo,
        solution_cost,
        search_time,
        nb_iter,
        validation,
        validation_details,
) -> None:
    file_path = 'nsp_benchmark.csv'
    fieldnames = [
        'date',
        'algo',
        'search_time',
        'nb_iter',
        'solution_cost',
        'validation',
        'problem_params',
        'algo_params',
        'validation_details',
        'commit',
    ]
    entry = {
        'date': get_date_time(),
        'algo': algo.__class__.__name__,
        'search_time': search_time,
        'nb_iter': nb_iter,
        'solution_cost': solution_cost,
        'validation': validation,
        'problem_params': problem.__dict__,
        'algo_params': algo.__dict__,
        'validation_details': validation_details,
        'commit': get_git_revision_hash(),
    }
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(entry)
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            print('creating the file')
            with open('nsp_benchmark.csv', 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(entry)
        else:
            raise e


def get_date_time() -> str:
    return datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

# benchmark_track(
#     problem=problem,
#     algo=algo,
#     solution_cost=0,
#     search_time=18,
#     nb_iter=9,
#     overall_validation=True,
# )
