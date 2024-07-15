import numpy as np
import scipy.optimize as spo
from battery_optimizer import BatteryOptimizer


# Child opimizer class for optimizing battery size for most cost-effective operation
class PriceOptimizer(BatteryOptimizer):
    def __init__(
        self,
        pred_net_load,
        duration,
        batt_price_per_kWh,
        import_tariff,
        batt_life,
        soc_min,
        soc_max,
        charge_efficiency,
        discharge_efficiency,
        self_dis,
        timestep_size,
        max_charge_rate,
        max_discharge_rate,
    ):
        super().__init__(
            pred_net_load,
            duration,
            batt_price_per_kWh,
            import_tariff,
            batt_life,
            soc_min,
            soc_max,
            charge_efficiency,
            discharge_efficiency,
            self_dis,
            timestep_size,
            max_charge_rate,
            max_discharge_rate,
        )
        self.x0 = np.zeros(pred_net_load.shape[0] + 1)

    def get_objective(self):
        def objective(x):
            # Calculate the cost of the battery assuming 10 year life span
            batt_price_for_duration = (
                self.batt_price_per_kWh
                * x[-1]
                * (self.duration / (365 * self.batt_life))
            )
            sum = 0
            for i in range(x.shape[0] - 1):
                # Assuming no remuneration
                energy = max(0, x[i] * x[-1] + self.pred_net_load[i])
                sum += energy * self.import_tariff[i]
            return sum + batt_price_for_duration

        return objective

    def get_constraints(self):
        return [
            spo.NonlinearConstraint(
                lambda x: self.calc_soc(x[:-1]), self.soc_min, self.soc_max
            )
        ]
    
    def get_bounds(self):
        max_discharge_percent = -1.0 * self.max_discharge_rate * self.timestep_size
        max_charge_percent = self.max_charge_rate * self.timestep_size
        bounds = [(max_discharge_percent, max_charge_percent) for _ in range(self.pred_net_load.shape[0])]
        bounds.append((0, 10000))
        return bounds

    