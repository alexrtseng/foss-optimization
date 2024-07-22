from price_optimizer import PriceOptimizer

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def optimize(week=True, **kwargs):
    df = pd.read_csv('data/Data_Households_SmartPV.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    if week:
        avg_df = df.groupby([df['Timestamp'].dt.day_of_week, df['Timestamp'].dt.hour]).mean()
        timestamps = pd.date_range(start='2016-06-06 00:00:00', end='2016-06-12 23:30:00', freq='1h')
        avg_df['Timestamp'] = timestamps
        avg_df.set_index('Timestamp', inplace=True)
        avg_df.index = pd.to_datetime(avg_df.index)
    else:    
        avg_df = df.groupby([df['Timestamp'].dt.day_of_year, df['Timestamp'].dt.hour]).mean()
        avg_df['Timestamp'] = avg_df['Timestamp'].dt.round('h')
        avg_df.set_index('Timestamp', inplace=True)

    avg_df = avg_df / 1000 # Convert to kW

    # Load tariffs
    import_tariff = pd.read_csv("tariffs/import_tariff.csv")
    import_tariff = import_tariff["import_tariff"].values

    export_tariff = pd.read_csv("tariffs/export_tariff.csv")
    export_tariff = export_tariff["export_tariff"].values

    # Load battery parameters
    if 'batt_price_per_kWh' in kwargs:
        batt_price_per_kWh = kwargs['batt_price_per_kWh']
    else:
        batt_price_per_kWh = 1300
    if 'batt_life' in kwargs:
        batt_life = kwargs['batt_life']
    else:
        batt_life = 10
    soc_min = 0.1
    soc_max = 0.9
    charge_efficiency = 0.98
    discharge_efficiency = 0.98
    self_dis = 0.02
    timestep_size = 1
    max_charge_rate = 0.5
    max_discharge_rate = 0.5

    # Create optimizer
    price_optimizer = PriceOptimizer(
        pred_net_load=avg_df['NetLoad'],
        duration=(7 if week else 366), # Leap year
        batt_price_per_kWh=batt_price_per_kWh,
        import_tariff=import_tariff,
        export_tariff=export_tariff,
        batt_life=batt_life,
        soc_min=soc_min,
        soc_max=soc_max,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        self_dis=self_dis,
        timestep_size=timestep_size,
        max_charge_rate=max_charge_rate,
        max_discharge_rate=max_discharge_rate,
    )

    # Optimize
    start_time = time.time()
    result = price_optimizer.local_optimize()
    end_time = time.time()
    execution_time = end_time - start_time

    # Calculate price without battery
    no_bat_price = price_optimizer.get_objective()(np.zeros(price_optimizer.pred_net_load.shape[0] + 1))
    if week:
        yearly_savings = (no_bat_price - result.fun) * 52
    else:
        yearly_savings = (no_bat_price - result.fun)

    # Print results
    print(result)
    print(f"Execution time: {execution_time}")

    # Dump results to csv
    if 'version_notes' in kwargs:
        version_notes = kwargs['version_notes']
    else:
        version_notes = 'N/A'
    with open('version/version_num.txt', 'r+') as file:
        version = int(file.read())
        file.seek(0)
        file.write(str(version + 1))
    
    result_df = avg_df.drop(columns=['NM_meter', 'Import', 'Export', 'PV_meter', 'Production', 'Consumption'])
    result_df['OptimalCharge'] = result.x[:-1]
    result_df['OptimalSOC'] = price_optimizer.calc_soc(result.x[:-1])[1:]
    result_df.to_csv(f"results/result_df_v{version}.csv", index=False)
    
    # Write parameters to file
    with open(f"version/optimize_versions.txt", 'a') as file:
        file.write(f"Version: {version}\n")
        file.write(f"Version notes:\n{version_notes}\n")
        file.write(f"batt_price_per_kWh: {batt_price_per_kWh}\n")
        file.write(f"batt_life: {batt_life}\n")
        file.write(f"soc_min: {soc_min}\n")
        file.write(f"soc_max: {soc_max}\n")
        file.write(f"charge_efficiency: {charge_efficiency}\n")
        file.write(f"discharge_efficiency: {discharge_efficiency}\n")
        file.write(f"self_dis: {self_dis}\n")
        file.write(f"timestep_size: {timestep_size}\n")
        file.write(f"max_charge_rate: {max_charge_rate}\n")
        file.write(f"max_discharge_rate: {max_discharge_rate}\n\n")
        file.write("Results:\n")
        file.write(f"Execution time: {execution_time}s\n")
        file.write(f"Optimization Duration: {'Week' if week else 'Year'}\n")
        file.write(f"Price without battery: {no_bat_price}\n")
        file.write(f"Optimal battery capacity: {result.x[-1]}kWh\n")
        file.write(f"Final objective value (price): {result.fun}\n\n")
        file.write(f"Yearly savings: {yearly_savings}\n")
        file.write("________________________________________________________\n\n")

    # Graph result for random day, week, month
    result_df.plot(y=['NetLoad', 'OptimalSOC'], title='Example Week Optimal Charge', figsize=(12, 6))
    plt.show()
    

if __name__ == "__main__":
    version_notes = 'Realistic Bat price CPP ($0.8 5-8pm; $0.2 otherwise) no export remuneration'
    optimize(week=True, version_notes=version_notes, batt_price_per_kWh=1300)