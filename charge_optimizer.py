import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import cvxpy as cp

from battery_optimizer import BatteryOptimizer


# Class for optimizer without cost consideration; just battery control
class ChargeOptimizer(BatteryOptimizer):
    def __init__(
        self,
        pred_net_load,
        duration,  # Duration of time (in days) the alg. is optimizing for
        import_tariff: np.array,  # array of import tariffs for each hour
        export_tariff: np.array,  # array of export tariffs for each hour
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
            export_tariff,
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

    def get_x0(self):
        return np.zeros(self.pred_net_load.shape[0])

    def get_objective(self):

        def objective(x):
            sum = 0
            for i in range(x.shape[0]):
                net_energy = (x[i] * self.batt_capacity) + self.pred_net_load.iloc[i]
                if net_energy >= 0:
                    sum += (net_energy * self.import_tariff[i])  
                else: 
                    sum += net_energy * self.export_tariff[i]
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
    
    def convex_optimize(self, batt_capacity):
        rt_efficiency = self.charge_efficiency * self.discharge_efficiency

        charge = cp.Variable(self.x0.shape[0])
        discharge = cp.Variable(self.x0.shape[0])

        # Penalty for charging and discharging at the same time
        penalty = (cp.sum(charge) + cp.sum(discharge)) * 0.0001
        objective = cp.Minimize(cp.sum(
            cp.multiply(
                (charge * batt_capacity) + self.pred_net_load, self.import_tariff)
        ) + cp.sum(
            cp.multiply(
                (-1 * discharge * batt_capacity * rt_efficiency) + self.pred_net_load, self.import_tariff)
        ) + penalty
        )

        constraints = [
            charge >= 0,
            discharge >= 0,
            charge <= self.max_charge_rate * self.timestep_size,
            discharge <= self.max_discharge_rate * self.timestep_size,
            cp.cumsum(charge - discharge) <= batt_capacity * self.soc_max,
            cp.cumsum(charge - discharge) >= batt_capacity * self.soc_min,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal var", charge.value, discharge.value)
        
        return prob, charge, discharge

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
