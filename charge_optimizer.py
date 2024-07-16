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
        if duration.is_integer() and duration >= 1:
            guess = np.array([ 0.00255102,  0.00255102,  0.00255102,  0.00255102,  0.00255102,
                0.00255102,  0.00255102,  0.00255102,  0.00255102,  0.00255102,
                0.00255102,  0.00255102,  0.00255102,  0.00255102,  0.00255102,
                0.00255102,  0.00255102,  0.00255102,  0.02413543,  0.04968567,
                0.07157302,  0.08947047,  0.10304017,  0.11190971,  0.11581524,
                0.11362456,  0.10800467,  0.10329501,  0.09237772,  0.07319358,
                0.05486604,  0.03685703,  0.01559768, -0.00571888, -0.02624013,
                -0.04702403, -0.06654517, -0.08275002, -0.09311118, -0.10355396,
                -0.11098244, -0.00292598,  0.00255102,  0.00255102,  0.00255102,
                0.00255102,  0.00255102,  0.00255102
            ])
            self.x0 = np.tile(guess, int(duration))
        else:
            self.x0 = np.zeros(pred_net_load.shape[0])

    def set_batt_capacity(self, batt_capacity):
        self.batt_capacity = batt_capacity

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
    
    def l_bfgs_b_optimize(self, exp_func_params=(0.65, 79), maxiter=3000, disp=True, maxfun=50000):
        bounds = self.get_bounds()
        objective = self.get_objective()
        def objective_with_constraints(x):
            soc = self.calc_soc(x)
            soc_lb = exp_func_params[0] * np.exp(-exp_func_params[1] * (soc - self.soc_min))
            soc_ub = exp_func_params[0] * np.exp(exp_func_params[1] * (soc - self.soc_max))

            return objective(x) + np.sum(soc_lb) + np.sum(soc_ub)

        l_bfgs_b_x0 = self.x0
        l_bfgs_b_x0[0] = 0.4
        result = spo.minimize(
            objective_with_constraints,
            self.x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={"maxiter": maxiter, 'disp': disp, 'maxfun': maxfun}
        )
        self.result = result
        return result

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

    