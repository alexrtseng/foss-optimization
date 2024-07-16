from matplotlib.pylab import rand
from pandas import options
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt


# Super class for capacity optimizers
class BatteryOptimizer:
    def __init__(
        self,
        pred_net_load,
        duration,
        batt_price_per_kWh,
        import_tariff: np.array,  # array of import tariffs; can be for each point or each hour
        batt_life,  # Battery life in years
        soc_min,
        soc_max,
        charge_efficiency,
        discharge_efficiency,
        self_dis,
        timestep_size,  # Duration between each timestep in hours
        max_charge_rate,  # Percentage of battery capacity that can be charged in one hour
        max_discharge_rate,
    ):
        self.pred_net_load = pred_net_load
        self.duration = duration
        self.batt_price_per_kWh = batt_price_per_kWh
        self.import_tariff = import_tariff
        self.batt_life = batt_life
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.self_discharge = self_dis
        self.timestep_size = timestep_size
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.result = None
        self.x0 = None  # Child classes should set this value
        if import_tariff.shape[0] == 24:
            steps_in_hour = (1.0 / timestep_size)
            assert steps_in_hour.is_integer()
            day_array = np.repeat(import_tariff, steps_in_hour)
            self.import_tariff = np.tile(day_array, duration)
        else:
            self.import_tariff = import_tariff


    def get_objective(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_constraints(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_bounds(self):
        raise NotImplementedError("Subclasses must implement this method")

    def calc_soc(self, charge_vals):
        soc = np.zeros(charge_vals.shape[0] + 1)
        soc[0] = self.soc_min
        for t in range(charge_vals.shape[0]):
            soc[t + 1] = (
                soc[t]
                + (
                    (
                        (charge_vals[t] * self.charge_efficiency)
                        if charge_vals[t] > 0
                        else (charge_vals[t] / self.discharge_efficiency)
                    )
                )
                - (self.self_discharge * soc[t])
            )

        return soc

    # Perform the optimization; duration is optimized days (in days)
    def local_optimize(self, method: str = "SLSQP", disp=True, tol=1e-10, maxiter=4000):
        bounds = self.get_bounds()
        result = spo.minimize(
            self.get_objective(),
            self.x0,
            bounds=bounds,
            method=method,
            constraints=self.get_constraints(),
            options={"maxiter": maxiter, "disp": disp},
            tol=tol,
        )
        self.result = result
        return result

    # Perform the optimization globally; duration is optimized days (in days)
    def global_optimize(self):
        objective_func = self.get_objective()
        bounds = self.get_bounds()
        constraints = self.get_constraints()

        result = spo.differential_evolution(
            func=objective_func,
            bounds=bounds,
            constraints=constraints,
            maxiter=5000,
            popsize=15,
            strategy="best1bin",
        )
        self.result = result
        return result
