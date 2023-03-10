import statistics


class IterNext:
    def __init__(
            self,
            nb_iter_max: int = 100,
            zero_cost: bool = True,
            validation: bool = True,
            zero_cost_max: int = 0,
            mean_hist_cost: list[int, float] = [0, 1.0],
            check_every: int = 1,
    ) -> None:
        self.zero_cost = zero_cost
        self.validation = validation
        self.mean_hist_cost = mean_hist_cost
        self.nb_iter_max = nb_iter_max
        self.check_every = check_every
        self.zero_cost_max = zero_cost_max

    def check_if_one_more_iter(
            self,
            iter: int,
            covering_cost: int,
            validate: bool,
            states: list,
    ) -> bool:
        """
        Function that returns True if the algorithm should continue to iterate
        and False if it should stop.
        The criterias are, in this order:
        - If the number of iterations is greater than nb_iter_max, the algorithm
        will stop
        - If zero_cost is True, the algorithm will only stop if the covering 
        cost is 0
        - If validation is True, the algorithm will only stop if the solution is 
        valid
        - If zero_cost_max is greater than 0, the algorithm will stop if the
        covering cost is 0 for zero_cost_max iterations
        - If mean_hist_cost[0] is greater than 0, the algorithm will
        stop if the covering_cost is within mean_hist_cost[1] percentage of the
        historical covering cost rolling average of the last mean_hist_cost[0]
        iterations
        Checks are done every check_every iterations
        """
        if iter > self.nb_iter_max:
            return False
        if iter % self.check_every != 0:
            return True
        if self.validation and not validate or self.zero_cost and covering_cost != 0:
            return True
        if self.zero_cost_max > 0 and covering_cost == 0:
            test_space = states[-self.zero_cost_max:]
            if len(test_space) == self.zero_cost_max and \
                    all([cost == 0 for cost in test_space]):
                return False
        if self.mean_hist_cost[0] > 0:
            test_space = states[-self.mean_hist_cost[0]:]
            if len(test_space) == self.mean_hist_cost[0] and \
                    self.diff_percent(covering_cost, statistics.fmean(test_space)) < self.mean_hist_cost[1]:
                return False

    def diff_percent(
        self,
            a: float,
            b: float,
    ) -> float:
        """
        Returns the percentage difference between a and b, based on b
        """
        dist = (a - b) ** 2 ** 0.5
        diff_percent = dist / b
        return diff_percent