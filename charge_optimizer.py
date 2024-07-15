import numpy as np
import scipy.optimize as spo

from .capacity_optimizer import BatteryOptimizer


# Class for optimizer without cost consideration; just battery control
class ChargeOptimizer(BatteryOptimizer):
    def __init__(
        self,
        pred_net_load,
        duration,
        import_tariff,
        soc_min,
        soc_max,
        charge_efficiency,
        discharge_efficiency,
        self_dis,
        timestep_size,
        max_charge_rate,
        max_discharge_rate,
        batt_capacity,
    ):
        self.batt_capacity = batt_capacity
        super().__init__(
            pred_net_load,
            duration,
            0,
            import_tariff,
            soc_min,
            soc_max,
            charge_efficiency,
            discharge_efficiency,
            self_dis,
            timestep_size,
            max_charge_rate,
            max_discharge_rate,
        )

    def get_objective(self):

        def objective(x):
            sum = 0
            for i in range(x.shape[0]):
                sum += max(
                    0,
                    (
                        (x[i] * self.batt_capacity * self.discharge_efficiency)
                        + self.pred_net_load[i]
                    ),
                )
            return sum
        
        return objective
    
    def get_constraints(self):
        return [spo.NonlinearConstraint(lambda x: self.calc_soc(x), self.soc_min, self.soc_max)]

    def get_bounds(self):
        max_discharge_percent = -1.0 * self.max_discharge_rate * self.timestep_size
        max_charge_percent = self.max_charge_rate * self.timestep_size
        return [(max_discharge_percent, max_charge_percent) for _ in range(self.pred_net_load.shape[0])]
