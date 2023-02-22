

if __name__ == '__main__':
    # parameters
    nb_nurses = 4
    nb_work_days_per_week = 7
    nb_shifts_per_work_day = 1
    nb_nrs_per_shift = 2
    nrs_max_work_days_per_week = 5

    # generate the set of possible solutions
    solution_set = generate_solution_set(nb_nurses, nb_shifts, max_work_days)

    # create the set of possible shifts, and associated penalty costs

    search_solution()