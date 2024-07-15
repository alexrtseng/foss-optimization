import numpy as np
import scipy.optimize as spo

from battery_optimizer import BatteryOptimizer


# Class for optimizer without cost consideration; just battery control
class ChargeOptimizer(BatteryOptimizer):
    def __init__(
        self,
        pred_net_load,
        duration,  # Duration of time (in days) the alg. is optimizing for
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
        super().__init__(
            pred_net_load,
            duration,
            0,
            0,
            0,
            soc_min,
            soc_max,
            charge_efficiency,
            discharge_efficiency,
            self_dis,
            timestep_size,
            max_charge_rate,
            max_discharge_rate,
        )
        self.batt_capacity = batt_capacity
        self.x0 = np.zeros(pred_net_load.shape[0])


    def get_objective(self):

        def objective(x):
            sum = 0
            for i in range(x.shape[0]):
                sum += max(
                    0,
                    (
                        (x[i] * self.batt_capacity)
                        + self.pred_net_load.iloc[i]
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
