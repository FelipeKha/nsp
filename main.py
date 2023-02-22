import matplotlib.pyplot as plt

from tabu_search import TabuSearch
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

    # run tabu search
    tabu_search = TabuSearch(
        nb_nurses,
        nb_work_days_per_week,
        nb_shifts_per_work_day,
        nb_nrs_per_shift,
        nrs_max_work_days_per_week,
        nb_iter,
        tabu_limit,
    )
    
    solution, solution_cost, states = tabu_search.search_solution()
    
    # print results
    print('solution:', solution)
    print('solution_cost:', solution_cost)

    plt.plot(states)
    plt.xlabel('nb iterations')
    plt.ylabel('coverage cost')
    plt.show()