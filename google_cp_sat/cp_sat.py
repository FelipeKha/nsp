"""
Example of a simple nurse scheduling problem.
Source:
https://developers.google.com/optimization/scheduling/employee_scheduling
"""
from ortools.sat.python import cp_model
import numpy as np


class CPSAT:
    def __init__(
            self,
            num_nurses: int,
            num_days: int,
            num_shifts: int,
            target_nurses_per_shift: int,
            max_work_days_per_week: int,
    ) -> None:
        self.num_nurses = num_nurses
        self.num_days = num_days
        self.num_shifts = num_shifts
        self.target_nurses_per_shift = target_nurses_per_shift
        self.max_work_days_per_week = max_work_days_per_week

    def build_model(self):
        # Creates the model.
        model = cp_model.CpModel()

        # Creates shift variables.
        # shifts[(n, d, s)]: nurse 'n' works shift 's' on day 'd'.
        shifts = {}
        for n in range(self.num_nurses):
            for d in range(self.num_days):
                for s in range(self.num_shifts):
                    shifts[(n, d,
                            s)] = model.NewBoolVar('shift_n%id%is%i' % (n, d, s))

        # Each shift is assigned to exactly one nurse in the schedule period.
        for d in range(self.num_days):
            for s in range(self.num_shifts):
                model.Add(sum(shifts[(n, d, s)] for n in range(self.num_nurses)) == self.target_nurses_per_shift)

        # Each nurse works at most one shift per day.
        for n in range(self.num_nurses):
            for d in range(self.num_days):
                model.AddAtMostOne(shifts[(n, d, s)] for s in range(self.num_shifts))

        # Each nurse works at most 5 days per week.
        for n in range(self.num_nurses):
            model.Add(sum(sum(shifts[(n, d, s)] for s in range(self.num_shifts)) for d in range(self.num_days)) <= self.max_work_days_per_week)

        # Try to distribute the shifts evenly, so that each nurse works
        # min_shifts_per_nurse shifts. If this is not possible, because the total
        # number of shifts is not divisible by the number of nurses, some nurses will
        # be assigned one more shift.
        min_shifts_per_nurse = (self.num_shifts * self.num_days * self.target_nurses_per_shift) // self.num_nurses
        if self.num_shifts * self.num_days * self.target_nurses_per_shift % self.num_nurses == 0:
            max_shifts_per_nurse = min_shifts_per_nurse
        else:
            max_shifts_per_nurse = min_shifts_per_nurse + 1
        for n in range(self.num_nurses):
            shifts_worked = []
            for d in range(self.num_days):
                for s in range(self.num_shifts):
                    shifts_worked.append(shifts[(n, d, s)])
            model.Add(min_shifts_per_nurse <= sum(shifts_worked))
            model.Add(sum(shifts_worked) <= max_shifts_per_nurse)

        return model, shifts

    # not working
    def build_objective_function(self, model, shifts):
        sum_cov = []
        for s in range(self.num_shifts):
            shift_cover = sum(shifts[(n, d, s)] for d in range(self.num_days) for n in range(self.num_nurses))
            coverage = self.target_nurses_per_shift - shift_cover
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

    def search_solution(self):
        model, shifts = self.build_model()
        # model = self.build_objective_function(model, shifts)
        solver = self.build_solver()
        
        solution_limit = 1
        solution_formatter = CPSATSolutionFormatter(shifts, self.num_nurses,
                                                        self.num_days, self.num_shifts,
                                                        solution_limit)
        
        solver.Solve(model, solution_formatter)
        solution = solution_formatter.get_solution()
        # print('cost:', solver.ObjectiveValue())
        
        # # Statistics.
        # print('\nStatistics')
        # print('  - conflicts      : %i' % solver.NumConflicts())
        # print('  - branches       : %i' % solver.NumBranches())
        # print('  - wall time      : %f s' % solver.WallTime())

        return solution, 0, [0]


class CPSATSolutionFormatter(cp_model.CpSolverSolutionCallback):
    def __init__(self, shifts, num_nurses, num_days, num_shifts, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._shifts = shifts
        self._num_nurses = num_nurses
        self._num_days = num_days
        self._num_shifts = num_shifts
        self._solution_count = 0
        self._solution_limit = limit
        self.solution = np.zeros((self._num_nurses, self._num_days * self._num_shifts))

    def on_solution_callback(self):
        for n in range(self._num_nurses):
            for d in range(self._num_days):
                for s in range(self._num_shifts):
                    if self.Value(self._shifts[(n, d, s)]):
                        self.solution[n, d] = 1
                    else: 
                        self.solution[n, d] = 0
        self.StopSearch()

    def get_solution(self):
        return self.solution
    

# if __name__ == '__main__':
#     cpsat = CPSAT(4, 7, 1, 2, 5)
#     solution, _, _ = cpsat.search_solution()
#     print('solution received:', solution)