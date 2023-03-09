"""
Example of a simple nurse scheduling problem.
Source:
https://developers.google.com/optimization/scheduling/employee_scheduling
"""
from ortools.sat.python import cp_model
import numpy as np

from problem_setup.problem import Problem


class CPSAT:
    def __call__(
            self,
            problem: Problem,
    ) -> None:
        model, shifts = self.build_model(problem)
        # model = self.build_objective_function(model, shifts)
        solver = self.build_solver()

        solution_limit = 1
        solution_formatter = CPSATSolutionFormatter(
            shifts,
            problem,
            solution_limit
        )

        solver.Solve(model, solution_formatter)
        solution = solution_formatter.get_solution()
        # print('cost:', solver.ObjectiveValue())

        # # Statistics.
        # print('\nStatistics')
        # print('  - conflicts      : %i' % solver.NumConflicts())
        # print('  - branches       : %i' % solver.NumBranches())
        # print('  - wall time      : %f s' % solver.WallTime())

        return solution, 0, [0]

    def build_model(self, problem: Problem) -> tuple[cp_model.CpModel, dict]:
        # Creates the model.
        model = cp_model.CpModel()

        # Creates shift variables.
        # shifts[(n, d, s)]: nurse 'n' works shift 's' on day 'd'.
        shifts = {}
        for n in range(problem.nb_nurses):
            for d in range(problem.nb_work_days_per_week):
                for s in range(problem.nb_shifts_per_work_day):
                    shifts[(n, d,
                            s)] = model.NewBoolVar('shift_n%id%is%i' % (n, d, s))

        # Each shift is assigned to exactly one nurse in the schedule period.
        for d in range(problem.nb_work_days_per_week):
            for s in range(problem.nb_shifts_per_work_day):
                model.Add(sum(shifts[(n, d, s)] for n in range(
                    problem.nb_nurses)) == problem.target_nb_nrs_per_shift)

        # Each nurse works at most one shift per day.
        for n in range(problem.nb_nurses):
            for d in range(problem.nb_work_days_per_week):
                model.AddAtMostOne(shifts[(n, d, s)]
                                   for s in range(problem.nb_shifts_per_work_day))

        # Each nurse works at most 5 days per week.
        for n in range(problem.nb_nurses):
            model.Add(sum(sum(shifts[(n, d, s)] for s in range(problem.nb_shifts_per_work_day)) for d in range(
                problem.nb_work_days_per_week)) <= problem.nrs_max_work_days_per_week)

        # Try to distribute the shifts evenly, so that each nurse works
        # min_shifts_per_nurse shifts. If this is not possible, because the total
        # number of shifts is not divisible by the number of nurses, some nurses will
        # be assigned one more shift.
        min_shifts_per_nurse = (problem.nb_shifts_per_work_day *
                                problem.nb_work_days_per_week * problem.target_nb_nrs_per_shift) // problem.nb_nurses
        if problem.nb_shifts_per_work_day * problem.nb_work_days_per_week * problem.target_nb_nrs_per_shift % problem.nb_nurses == 0:
            max_shifts_per_nurse = min_shifts_per_nurse
        else:
            max_shifts_per_nurse = min_shifts_per_nurse + 1
        for n in range(problem.nb_nurses):
            shifts_worked = []
            for d in range(problem.nb_work_days_per_week):
                for s in range(problem.nb_shifts_per_work_day):
                    shifts_worked.append(shifts[(n, d, s)])
            model.Add(min_shifts_per_nurse <= sum(shifts_worked))
            model.Add(sum(shifts_worked) <= max_shifts_per_nurse)

        return model, shifts

    # not working
    def build_objective_function(self, model, shifts, problem: Problem):
        sum_cov = []
        for s in range(problem.nb_shifts_per_work_day):
            shift_cover = sum(shifts[(n, d, s)] for d in range(
                problem.nb_work_days_per_week) for n in range(problem.nb_nurses))
            coverage = problem.target_nb_nrs_per_shift - shift_cover
            square_x = model.NewIntVar(0, 100, "square_x")
            model.AddMultiplicationEquality(square_x, [coverage, coverage])
            sum_cov.append(square_x)
        model.Minimize(sum(sum_cov))

        return model

    def build_solver(self):
        # Creates the solver and solve.
        solver = cp_model.CpSolver()
        solver.parameters.linearization_level = 0
        # Enumerate all solutions.
        solver.parameters.enumerate_all_solutions = True
        return solver

    # def search_solution(self):
    #     model, shifts = self.build_model()
    #     # model = self.build_objective_function(model, shifts)
    #     solver = self.build_solver()

    #     solution_limit = 1
    #     solution_formatter = CPSATSolutionFormatter(shifts, self.nb_nurses,
    #                                                 self.nb_work_days_per_week, self.nb_shifts_per_work_day,
    #                                                 solution_limit)

    #     solver.Solve(model, solution_formatter)
    #     solution = solution_formatter.get_solution()
    #     # print('cost:', solver.ObjectiveValue())

    #     # # Statistics.
    #     # print('\nStatistics')
    #     # print('  - conflicts      : %i' % solver.NumConflicts())
    #     # print('  - branches       : %i' % solver.NumBranches())
    #     # print('  - wall time      : %f s' % solver.WallTime())

    #     return solution, 0, [0]


class CPSATSolutionFormatter(cp_model.CpSolverSolutionCallback):
    def __init__(
            self,
            shifts: dict,
            problem: Problem,
            limit: int
    ) -> None:
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._shifts = shifts
        self.problem = problem
        self._solution_count = 0
        self._solution_limit = limit
        self.solution = np.zeros(
            (self.problem.nb_nurses, self.problem.nb_work_days_per_week * self.problem.nb_shifts_per_work_day))

    def on_solution_callback(self) -> None:
        for n in range(self.problem.nb_nurses):
            for d in range(self.problem.nb_work_days_per_week):
                for s in range(self.problem.nb_shifts_per_work_day):
                    if self.Value(self._shifts[(n, d, s)]):
                        self.solution[n, d] = 1
                    else:
                        self.solution[n, d] = 0
        self.StopSearch()

    def get_solution(self) -> np.ndarray:
        return self.solution


# if __name__ == '__main__':
#     cpsat = CPSAT(4, 7, 1, 2, 5)
#     solution, _, _ = cpsat.search_solution()
#     print('solution received:', solution)
