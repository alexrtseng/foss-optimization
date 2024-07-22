import numpy as np
import scipy.optimize as spo
from battery_optimizer import BatteryOptimizer
import matplotlib.pyplot as plt
from charge_optimizer import ChargeOptimizer

# Child opimizer class for optimizing battery size for most cost-effective operation
class PriceOptimizer(BatteryOptimizer):
    def __init__(
        self,
        pred_net_load,
        duration,
        batt_price_per_kWh,
        import_tariff,
        export_tariff,
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
            export_tariff,
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
        self.x0[-1] = 2
        self.batt_ub = 100
        self.batt_lb = 0

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
                net_energy = (x[i] * x[-1]) + self.pred_net_load.iloc[i]
                if net_energy >= 0:
                    sum += (net_energy * self.import_tariff[i])  
                else: 
                    sum += net_energy * self.export_tariff[i]
            return sum + batt_price_for_duration

        return objective
    
    def get_x0(self):
        x0 = np.zeros(self.pred_net_load.shape[0] + 1)
        x0[-1] = 2
        return x0

    def get_constraints(self):
        return [
            spo.NonlinearConstraint(
                lambda x: self.calc_soc(x[:-1]), self.soc_min, self.soc_max
            )
        ]

    def get_bounds(self):
        max_discharge_percent = -1.0 * self.max_discharge_rate * self.timestep_size
        max_charge_percent = self.max_charge_rate * self.timestep_size
        bounds = [
            (max_discharge_percent, max_charge_percent)
            for _ in range(self.pred_net_load.shape[0])
        ]
        bounds.append((self.batt_lb, self.batt_ub))
        return bounds

    # Use bilevel optimization to optimize battery size; inner is charge optimizer outer is batt capacity
    def bilevel_optimize(self, method: str = "BFGS", inner_method = 'local_optimize'):
        # Make ChargeOptimizer to function as inner optimizer
        charge_optimizer = ChargeOptimizer(
            pred_net_load=self.pred_net_load,
            duration=self.duration,
            import_tariff=self.import_tariff,
            export_tariff=self.export_tariff,
            soc_min=self.soc_min,
            soc_max=self.soc_max,
            charge_efficiency=self.charge_efficiency,
            discharge_efficiency=self.discharge_efficiency,
            self_dis=self.self_discharge,
            timestep_size=self.timestep_size,
            max_charge_rate=self.max_charge_rate,
            max_discharge_rate=self.max_discharge_rate,
            batt_capacity=0,
        )
        # Used to prevent negative values with smoothed function
        zero_fun = charge_optimizer.local_optimize(disp=False).fun

        def objective(x):
            batt_price_for_duration = (
                self.batt_price_per_kWh
                * x[-1]
                * (self.duration / (365 * self.batt_life))
            )
            charge_optimizer.set_batt_capacity(x[0])
        
            if inner_method == 'local_optimize':
                charge_optimizer.local_optimize(disp=False, tol=1e-6)
                val = charge_optimizer.result.fun
            elif inner_method == 'l_bfgs_b_optimize':
                charge_optimizer.l_bfgs_b_optimize(disp=False)
                val = charge_optimizer.final_objective()
            else:
                raise ValueError('Invalid inner method')
            if x[-1] >= 0:
                price = val + batt_price_for_duration
            else:
                price = zero_fun + (batt_price_for_duration * batt_price_for_duration)
            print(f'Iteration of battery capacity: {x[0]}. Objective value: {price}')
            return price

        # Outer optimization
        result = spo.minimize(
            objective,
            np.array([2]),
            method=method,
            options={"maxiter": 1000, "disp": True, 'eps': 2},
        )
        print(f'result: {result.x[0]}')
        batt_size = result.x[0]
        charge_optimizer.set_batt_capacity(batt_size)
        result = charge_optimizer.local_optimize(disp=False)
        result.x = np.append(result.x, batt_size)
        self.result = result
        return result

    def plot_results(self, x_vals, title, aggregate=True):
        soc = self.calc_soc(self.result.x[:-1]) * self.result.x[-1]
        if aggregate:
            aggregate = [0]
            for num in self.pred_net_load:
                aggregate.append(aggregate[-1] + num)

            plt.plot(
                x_vals,
                aggregate[1:],
                label="Aggregate Net Load",
                linestyle="dotted",
                color="grey",
            )
            
        plt.plot(
            x_vals,
            self.pred_net_load,
            label="Net Load (kW)",
            color="grey",
            linestyle="dashed",
        )
        plt.plot(x_vals, soc[1:], label="Battery SOC")
        plt.xlabel("Timestamp")
        plt.ylabel("Energy (kWh)")
        plt.title(title)
        plt.legend(loc="upper left")
        plt.xticks(rotation=45)
        plt.show()

