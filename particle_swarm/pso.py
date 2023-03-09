import numpy as np

from problem_setup.problem import Problem


class ParticleSwarmOptimization:
    def __init__(
            self,
            swarm_size: int,
            max_iter: int,
            c1: float,
            c2: float,
            w: float,
            alpha: float,
            get_initial_population: callable,
            covering_cost: callable,
            check_population_for_max_days_per_week: callable,
    ) -> None:
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.alpha = alpha
        self.get_initial_population = get_initial_population
        self.covering_cost = covering_cost
        self.check_population_for_max_days_per_week = (
            check_population_for_max_days_per_week
        )

    def __call__(
            self,
            problem: Problem,
    ) -> tuple[np.ndarray, int, list]:
        initial_swarm = \
            self.get_initial_population(self.swarm_size, problem)
        best_solution, best_cost, states = \
            self.particle_swarm_optimization(initial_swarm, problem)
        
        return best_solution, best_cost, states

    def get_pop_costs(
            self,
            swarm: np.ndarray,
            problem: Problem,
    ) -> np.ndarray:
        """
        Given a swarm of dim (swarm_size, nb_nurses, nb_shifts), returns the 
        cost for each particle of the swarm of dim (swarm_size)
        """
        costs = np.empty(self.swarm_size, dtype=int)
        for i in range(self.swarm_size):
            costs[i] = self.covering_cost(swarm[i], problem)
        return costs

    def ycompare_func(
            self,
            p: int,
            pb: int,
            gb: int,
    ) -> np.int64:
        """
        Given a shift p, historical best shift pb and historical best shift
        of the swarm gb, returns y value.
        y take value 1 if shift same as shift = gbest, -1 if shift = pbest, -1
        or 1 randomly if shift = pbest and shift = gbest, 0 otherwise.
        """
        out = (
            1 * (p == gb)
            + (-1) * (p == pb)
            + np.random.choice([-1, 1]) * (p == pb and p == gb)
        )
        return out
    
    def yupdate_func(
            self,
            ylambda: float,
    ) -> int:
        """
        Given a ylambda value, returns yupdate value.
        yupdate take value 1 if ylambda > alpha, -1 if ylambda < alpha, 0 
        otherwise.
        """
        out = (
            1 * (ylambda > self.alpha)
            + (-1) * (ylambda < self.alpha)
            + 0
        )
        return out
    
    def swarm_update_func(
            self,
            y: int,
            pb: int,
            gb: int,
    ) -> int:
        """
        Given y, pb and gb, returns:
        - pb if y = -1
        - gb if y = 1
        - random value 1 or 0 if y = 0
        """
        out = (
            pb * (y == -1)
            + gb * (y == 1)
            + np.random.choice([0, 1]) * (y == 0)
        )
        return out

    def get_ycompare(
            self,
            pcurrent: np.ndarray,
            pbest: np.ndarray,
            gbest: np.ndarray,
    ) -> np.ndarray:
        """
        Given current particles (swarm_size, nb_nurses, nb_shifts), historical 
        best for each particle (swarm_size, nb_nurses, nb_shifts) and historical 
        best particle of the swarm (nb_nurses, nb_shifts), returns y array of 
        dim (swarm_size, nb_nurses, nb_shifts).
        y take value 1 if shift same as shift = gbest, -1 if shift = pbest, -1 
        or 1 randomly if shift = pbest = gbest, 0 otherwise.
        """
        ycompare_func = np.vectorize(self.ycompare_func)
        ycompare = ycompare_func(
            pcurrent,
            pbest,
            gbest,
        )
        return ycompare

    def get_yupdate(
            self,
            ylambda: np.ndarray,
    ) -> np.ndarray:
        """
        Given ylambda array of dim (swarm_size, nb_nurses, nb_shifts), returns 
        yupdate array of dim (swarm_size, nb_nurses, nb_shifts).
        yupdate take value 1 if ylambda > alpha, -1 if ylambda < alpha, 0 
        otherwise.
        """
        yupdate_func = np.vectorize(self.yupdate_func)
        yupdate = yupdate_func(ylambda)
        return yupdate

    def get_d1_d2(
            self, 
            ycompare: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given y array of dim (swarm_size, nb_nurses, nb_shifts), returns d1 and 
        d2 arrays of dim (swarm_size, nb_nurses, nb_shifts).
        d1 distance between pcurrent and pbest (0 if pcurrrent = pbest).
        d2 distance between pcurrent and gbest (0 if pcurrrent = gbest).
        """
        d1 = -1 - ycompare
        d2 = 1 - ycompare
        return d1, d2

    def update_swarm(
            self,
            yupdate: np.ndarray,
            pbest: np.ndarray,
            gbest: np.ndarray,
    ) -> np.ndarray:
        """
        Given yupdate (swarm_size, nb_nurses, nb_shifts), pbest (swarm_size, 
        nb_nurses, nb_shifts) and gbest (nb_nurses, nb_shifts), returns updated 
        swarm.
        Particles of the updated swarm are updated according to the following:
        - if yupdate = 1, particle is updated with gbest
        - if yupdate = -1, particle is updated with pbest
        - if yupdate = 0, particle is updated to a random value
        """
        swarm_update_func = np.vectorize(self.swarm_update_func)
        swarm = swarm_update_func(
            yupdate,
            pbest,
            gbest,
        )
        return swarm

    def update_pbest(
            self,
            pcurrent: np.ndarray,
            pcurrent_costs: np.ndarray,
            pbest: np.ndarray,
            pbest_costs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given current particles (swarm_size, nb_nurses, nb_shifts), their costs
        (swarm_size), historical best for each particle (swarm_size, nb_nurses,
        nb_shifts) and their costs (swarm_size), returns updated best historical 
        particle and its cost.
        """
        for i in range(len(pcurrent)):
            if pcurrent_costs[i] < pbest_costs[i]:
                pbest[i] = pcurrent[i]
                pbest_costs[i] = pcurrent_costs[i]
        return pbest, pbest_costs

    def update_gbest(
            self,
            gbest: np.ndarray, 
            gbest_cost: int, 
            pbest: np.ndarray, 
            pbest_costs: np.ndarray,
    ) -> tuple[np.ndarray, np.int64]:
        """
        Given global best partile (nb_nurses, nb_shifts) and its cost, and 
        historical best for each particle (swarm_size, nb_nurses, nb_shifts) and 
        their costs (swarm_size), returns updated global best particle and its
        cost. 
        """
        index_gbest_candidate = np.argmin(pbest_costs)
        gbest_candidate_cost = pbest_costs[index_gbest_candidate]
        if gbest_candidate_cost < gbest_cost:
            gbest = pbest[index_gbest_candidate]
            gbest_cost = gbest_candidate_cost
        return gbest, gbest_cost

    def update_velocity(
            self,
            v: np.ndarray,
            d1: np.ndarray,
            d2: np.ndarray,
    ) -> np.ndarray:
        """
        Given velocity (swarm_size, nb_nurses, nb_shifts), current particles 
        (swarm_size, nb_nurses, nb_shifts), historical best for each particle 
        (swarm_size, nb_nurses, nb_shifts) and historical best particle of the
        swarm (nb_nurses, nb_shifts), returns the updated velocity of dim
        (swarm_size, nb_nurses, nb_shifts)
        """
        r1 = np.random.uniform(0, 1, size=v.shape)
        r2 = np.random.uniform(0, 1, size=v.shape)
        v = \
            self.w * v \
            + self.c1 * r1 * d1 \
            + self.c2 * r2 * d2
        return v

    def particle_swarm_optimization(
            self,
            initial_swarm: np.ndarray,
            problem: Problem,
    ) -> tuple[np.ndarray, int, list]:
        pcurrent = initial_swarm              # current swarm of particles (swarm_size, nb_nurses, nb_shifts)
        pcurrent_costs = self.get_pop_costs(pcurrent, problem) # costs of current particles (swarm_size)
        pbest = pcurrent                      # historical best particles for each particle (swarm_size, nb_nurses, nb_shifts)
        pbest_costs = pcurrent_costs
        gbest_index = np.argmin(pcurrent_costs)       # index of best particle of the swarm
        gbest = pbest[gbest_index]            # historical best particle of the swarm (nb_nurses, nb_shifts)
        gbest_cost = pcurrent_costs[gbest_index]
        v = np.random.uniform(-1, 1, size=pcurrent.shape) # velocity
        states = [gbest_cost]

        for i in range(self.max_iter):
            print(i)
            ycompare = self.get_ycompare(pcurrent, pbest, gbest)
            d1, d2 = self.get_d1_d2(ycompare)
            v = self.update_velocity(v, d1, d2)
            ylambda = ycompare + v
            yupdate = self.get_yupdate(ylambda)
            pcurrent = self.update_swarm(yupdate, pbest, gbest)
            pcurrent = self.check_population_for_max_days_per_week(
                pcurrent,
                problem,
            )
            pcurrent_costs = self.get_pop_costs(pcurrent, problem)
            pbest, pbest_costs = self.update_pbest(
                pcurrent, 
                pcurrent_costs, 
                pbest, 
                pbest_costs,
            )
            gbest, gbest_cost = self.update_gbest(
                gbest, 
                gbest_cost, 
                pbest, 
                pbest_costs,
            )
            states.append(gbest_cost)

        print('swarm:', pcurrent)
        return gbest, gbest_cost, states

    # def search_solution(self) -> tuple[np.ndarray, int, list]:
    #     initial_swarm = \
    #         self.get_population.get_initial_population(self.swarm_size)
    #     best_solution, best_cost, states = \
    #         self.particle_swarm_optimization(initial_swarm)
        
    #     return best_solution, best_cost, states
