import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt

from battery_optimizer import BatteryOptimizer


# Class for optimizer without cost consideration; just battery control
class ChargeOptimizer(BatteryOptimizer):
    def __init__(
        self,
        pred_net_load,
        duration,  # Duration of time (in days) the alg. is optimizing for
        import_tariff: np.array,  # array of import tariffs for each hour
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
            import_tariff,
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

    def set_batt_capacity(self, batt_capacity):
        self.batt_capacity = batt_capacity

    def set_new_input(self, pred_net_load, duration, import_tariff):
        self.pred_net_load = pred_net_load
        self.duration = duration
        self.x0 = np.zeros(pred_net_load.shape[0])
        if import_tariff.shape[0] == 24:
            steps_in_hour = (1.0 / self.timestep_size)
            assert steps_in_hour.is_integer()
            day_array = np.repeat(import_tariff, steps_in_hour)
            self.import_tariff = np.tile(day_array, duration)
        else:
            self.import_tariff = import_tariff

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
                ) * self.import_tariff[i]
            return sum
        
        return objective
    
    def get_constraints(self):
        return [spo.NonlinearConstraint(lambda x: self.calc_soc(x), self.soc_min, self.soc_max)]

    def get_bounds(self):
        max_discharge_percent = -1.0 * self.max_discharge_rate * self.timestep_size
        max_charge_percent = self.max_charge_rate * self.timestep_size
        return [(max_discharge_percent, max_charge_percent) for _ in range(self.pred_net_load.shape[0])]

    def plot_results(self, x_vals):
        soc = self.calc_soc(self.result.x)

        plt.plot(x_vals, self.pred_net_load, label='NetLoad')
        plt.plot(x_vals, soc[1:], label='SOC')
        plt.xlabel('Timestamp')
        plt.ylabel('Energy (kWh)')
        plt.title('NetLoad and SOC')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()