from tabu_search.tabu_search import TabuSearch

class TestTabuSearch:
    def test_get_random_nurse_schedule(self):
        tabu_search = TabuSearch(
            nb_nurses=4,
            nb_work_days_per_week=7,
            nb_shifts_per_work_day=1,
            nb_nrs_per_shift=2,
            nrs_max_work_days_per_week=5,
            nb_iter=1000,
            tabu_limit=10,
        )
        nurse_schedule = tabu_search.get_random_nurse_schedule()
        assert nurse_schedule.shape == (7,)
        assert nurse_schedule.sum() <= 5
        assert nurse_schedule.sum() >= 0