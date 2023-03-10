class Problem:
    def __init__(
        self,
        nb_nurses: int,
        nb_work_days_per_week: int,
        nb_shifts_per_work_day: int,
        target_nb_nrs_per_shift: int,
        nrs_max_work_days_per_week: int,
    ) -> None:
        self.nb_nurses = nb_nurses
        self.nb_work_days_per_week = nb_work_days_per_week
        self.nb_shifts_per_work_day = nb_shifts_per_work_day
        self.target_nb_nrs_per_shift = target_nb_nrs_per_shift
        self.nrs_max_work_days_per_week = nrs_max_work_days_per_week

    def __call__(self) -> tuple[int, int, int, int, int]:
        self.out = (
            self.nb_nurses,
            self.nb_work_days_per_week,
            self.nb_shifts_per_work_day,
            self.target_nb_nrs_per_shift,
            self.nrs_max_work_days_per_week,
        )
        return self.out
