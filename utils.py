import matplotlib.pyplot as plt
from pyomo.environ import SolverFactory, SolverStatus
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from param import *
from plotly.subplots import make_subplots

def check_res(res):
    eps = 1e-3  # 1% error tolerance
    for t in range(len(res.t)):
        if res.P_pv[t] > res.C_pv * res.P_pv_max[t] + eps:
            print(f'Error: P_pv > P_pv_max at time step {t}: [{res.P_pv[t]} > {res.P_pv_max[t] * res.C_pv}]')
        if res.P_pv[t] > res.P_nom_pv + eps:
            print(f'Error: P_pv > P_pv_nom at time step {t}: [{res.P_pv[t]} > {res.P_nom_pv}]')
        if res.P_discharge_bss[t] > res.P_nom_bss + eps:
            print(f'Error: P_discharge_bss > P_nom_bss at time step {t}: [{res.P_discharge_bss[t]} > {res.P_nom_bss}]')
        if res.P_charge_bss[t] > res.P_nom_bss + eps:
            print(f'Error: P_charge_bss  > P_nom_bss at time step {t}: [{res.P_charge_bss[t]} > {res.P_nom_bss}]')
        if res.P_discharge_ev[t] > res.P_nom_ev + eps:
            print(f'Error: P_discharge_ev > P_nom_ev at time step {t}: [{res.P_discharge_ev[t]} > {res.P_nom_ev}]')
        if res.P_charge_ev[t] > res.P_nom_ev + eps:
            print(f'Error: P_charge_ev > P_nom_ev at time step {t}: [{res.P_charge_ev[t]} > {res.P_nom_ev}]')
        if  (res.P_charge_ev[t] > eps or res.P_discharge_ev[t] > eps) and res.EV_connected[t] == 0:
            print(f'Error: (Dis)charging without EV connected at time step {t}: [{res.P_charge_ev[t]}|{res.P_discharge_ev[t]}]')
        if res.P_gen[t] > res.P_max_gen+eps:
            print(f'Error: P_gen > P_max_gen at time step {t}: [{res.P_gen[t]} > {res.P_max_gen}]')
        if res.SOC_bss[t] > SOC_max_bss * res.C_bss + eps:
            print(f'Error: SOC_bss > SOC_max_bss at time step {t}: [{res.SOC_bss[t]} > {SOC_max_bss * res.C_bss}]')
        if res.SOC_bss[t] < SOC_min_bss * res.C_bss - eps:
            print(f'Error: SOC_bss < SOC_min_bss at time step {t}: [{res.SOC_bss[t]} < {SOC_min_bss * res.C_bss}]')
        if res.SOC_ev[t] > SOC_max_ev * res.C_ev + eps:
            print(f'Error: SOC_ev > SOC_max_ev at time step {t}: [{res.SOC_ev[t]} > {SOC_max_ev * res.C_ev}]')
        if res.SOC_ev[t] < SOC_min_ev * res.C_ev - eps:
            print(f'Error: SOC_ev < SOC_min_ev at time step {t}: [{res.SOC_ev[t]} < {SOC_min_ev * res.C_ev}]')

        if res.P_imp[t] > eps and res.P_exp[t] > eps:
            print(f'Error: Import and export at the same time step {t}: [{res.P_imp[t]}, {res.P_exp[t]}]')

        if abs(res.P_imp[t] + res.P_pv[t] + res.P_gen[t] + res.P_discharge_bss[t] + res.P_discharge_ev[t] - res.P_charge_bss[t]- res.P_charge_ev[t] - res.P_load[t] - res.P_exp[t]) >= eps:
            print(f'Error: power balance offset ({res.P_imp[t] + res.P_pv[t] + res.P_gen[t]+res.P_discharge_bss[t]+res.P_discharge_ev[t] - res.P_charge_bss[t] - res.P_charge_ev[t] - res.P_load[t] - res.P_exp[t]}), at time step {t} \
                  [P_imp, P_pv, P_gen, P_bss, P_ev, P_load, P_exp]: [{res.P_imp[t]}, {res.P_pv[t]}, {res.P_gen[t]}, {res.P_charge_bss[t]-res.P_discharge_bss[t]}, {res.P_charge_ev[t]-res.P_discharge_ev[t]}, {res.P_load[t]}, {res.P_exp[t]}]')

    for t in res.t_dep:
        if res.SOC_ev[t] < SOC_target_ev - eps:
            print(f'Error: EV SOC < SOC_target_ev at time step {t}: [{res.SOC_ev[t]} < {SOC_target_ev}]')

    return

def print_res(res):
    print(f'Total operation cost : {365*delta_t*sum(res.P_gen*PI_gen + res.P_imp*PI_imp - res.P_exp*PI_exp)/res.n_days:.2f} \u20ac/year')
    print(f'Objective value : {res.objective:.2f} \u20ac')
    print(f'Total grid import : {delta_t*sum(res.P_imp):.2f} kWh for {delta_t*sum(res.P_imp)*PI_imp:.2f} \u20ac')
    print(f'Total grid export: {delta_t*sum(res.P_exp):.2f} kWh for {delta_t*sum(res.P_exp)*PI_exp:.2f} \u20ac')
    print(f'Total PV energy curtailed: {delta_t*sum(res.P_pv_max*res.C_pv - res.P_pv):.2f} kWh')

    print(f'Generator:')
    print(f'- Total fuel cost: {delta_t*PI_gen * sum(res.P_gen):.2f} \u20ac for {delta_t*sum(res.P_gen > 0):.2f} kWh')
    print(f'- Number of times the generator was turned on|off: {sum((res.P_gen[t] > 0) and (res.P_gen[t-1] == 0) for t in range(1, len(res.P_gen)))}|{sum((res.P_gen[t] == 0) and (res.P_gen[t-1] > 0) for t in range(1, len(res.P_gen)))}')
    print(f'- Average power of the generator when it is on: {sum(res.P_gen[res.P_gen>0])/(len(res.P_gen[res.P_gen>0]) if sum(res.P_gen)>0 else 1):.2f} kW')
    print(f'- Average power variation between two \'on\' timesteps: {sum(abs(res.P_gen[t] - res.P_gen[t-1]) for t in range(1, len(res.P_gen)) if res.P_gen[t] > 0 and res.P_gen[t-1] > 0)/(sum(res.P_gen > 0) if sum(res.P_gen)>0 else 1):.2f} kW')

    print(f'BSS:')
    print(f'- Final SOC of the battery: {res.SOC_bss[-1]/res.C_bss:.2f} %')
    print(f'- Total energy charged|discharged to the battery: {delta_t*sum(res.P_charge_bss):.2f}|{-delta_t*sum(res.P_discharge_bss):.2f} kWh')
    print(f'- Total energy lost through efficiency: {delta_t*sum(res.P_charge_bss+res.P_discharge_bss)*(1-eff_bss):.2f} kWh')

    print(f'EV:')
    print(f'- Final SOC of the EV: {res.SOC_ev[-1]/res.C_ev:.2f} %')
    print(f'- Total energy charged|discharged to the EV: {delta_t*sum(res.P_charge_ev):.2f}|{delta_t*sum(res.P_discharge_ev):.2f} kWh')
    print(f'- Total energy lost through efficiency: {delta_t*sum(res.P_charge_ev+res.P_discharge_ev)*(1-eff_ev):.2f} kWh')

    print(f'Energy balance:')
    print(f'- Total    Negative|Positive: {delta_t*(sum(res.P_load+res.P_exp+res.P_charge_bss+res.P_charge_ev)):.2f}|{delta_t*(sum(res.P_pv+res.P_gen+res.P_imp+res.P_discharge_bss+res.P_discharge_ev)):.2f} kWh')
    print(f'- Grid     Exported|Imported: {delta_t*sum(res.P_exp):.2f}|{delta_t*sum(res.P_imp):.2f} kWh')
    print(f'- Base         Load|PV      : {delta_t*sum(res.P_load):.2f}|{delta_t*sum(res.P_pv):.2f} kWh')
    print(f'- Bss        Stored|Released: {delta_t*sum(res.P_charge_bss):.2f}|{abs(delta_t*sum(res.P_discharge_bss)):.2f} kWh')
    print(f'- EV         Stored|Released: {delta_t*sum(res.P_charge_ev):.2f}|{abs(delta_t*sum(res.P_discharge_ev)):.2f} kWh')
    print(f'- Generator        |Produced:      |{delta_t*sum(res.P_gen):.2f} kWh')
    return

def plot_res(res):
    # Plot power values (PV, battery, EV, generator, demand) over time
    # These plots may be improved to show what you want
    fig, ax = plt.subplots(2, 1, figsize=(8, 4), dpi=180)
    # Plot first subplot with power references and demand curve
    color = 'tab:red'
    ax[0].set_xlabel('Time (hours)')
    ax[0].set_ylabel('Power [kW]', color=color)
    ax[0].plot(res.t, -1*res.P_exp, label='Exports')
    ax[0].plot(res.t, res.P_imp, label='Imports')
    ax[0].plot(res.t, -1*res.P_load, label='Load')
    ax[0].plot(res.t, res.P_charge_bss, label='BSS charge')
    ax[0].plot(res.t, res.P_charge_ev, label='EV charge')
    #ax[0].plot(res.t, res.P_pv_max, label='PV max.', linestyle=':')
    ax[0].plot(res.t, res.P_pv, label='PV')
    ax[0].plot(res.t, -res.P_discharge_bss, label='BSS discharge')
    ax[0].plot(res.t, -res.P_discharge_ev, label='EV discharge')
    ax[0].plot(res.t, res.P_gen, label='Generator')
    ax[0].tick_params(axis='y', labelcolor=color)
    ax[0].legend(ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    ax[0].grid(True)
    # Create a secondp subplot for EV and battery energy states
    color = 'tab:blue'
    ax[1].set_ylabel('State of charge [%]', color=color)
    ax[1].plot(res.t, res.SOC_ev/res.C_ev, label='EV', linestyle='--')
    ax[1].plot(res.t, res.SOC_bss/res.C_bss, label='Battery', linestyle='--')
    ax[1].tick_params(axis='y', labelcolor=color)
    ax[1].legend()#bbox_to_anchor=(0, .5))
    plt.grid(True)
    plt.show()
    return

def solve_model(m, res):
    # Solve the optimization problem
    solver = SolverFactory('gurobi')
    output = solver.solve(m)#, tee=True)  # Parameter 'tee=True' prints the solver output

    # Print elapsed time
    status = output.solver.status

    # Check the solution status
    if status == SolverStatus.ok:
        print("Simulation completed")
        res.save_results(m)
        res.save_sizing_results(m)
        return m, res
    elif status == SolverStatus.warning:
        print("Solver finished with a warning.")
    elif status == SolverStatus.error:
        print("Solver encountered an error and did not converge.")
    elif status == SolverStatus.aborted:
        print("Solver was aborted before completing the optimization.")
    else:
        print("Solver status unknown.")
    return None, None


def print_sizing_results(res):
    print(f"PV system size: {res.C_pv:.2f} kWp, inverter: {res.P_nom_pv:.2f} kW")
    print(f"Battery size: {res.C_bss:.2f} kWh, inverter: {res.P_nom_bss:.2f} kW")
    print(f"EV size: {res.C_ev:.2f} kWh, inverter: {res.P_nom_ev:.2f} kW")
    print(f"Generator size: {res.P_max_gen:.2f} kW")
    print(f"Total investment cost: {(PI_c_pv*res.C_pv + PI_c_bss*res.C_bss + PI_c_gen*res.P_max_gen + PI_c_inv*(res.P_nom_bss+res.P_nom_pv))/inv_hor:.2f} \u20ac/year")


def plot_res2(res, vert_lines=True):
    # Create an interactive plotly figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Power Values Over Time", "State of Charge Over Time"),
        specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
    )

    # Add traces for positive power values (generation)
    fig.add_trace(go.Scatter(x=res.datetime, y=res.P_imp, mode='lines', name='Imports', stackgroup='positive'), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.datetime, y=res.P_pv, mode='lines', name='PV', stackgroup='positive'), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.datetime, y=res.P_gen, mode='lines', name='Generator', stackgroup='positive'), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.datetime, y=res.P_discharge_bss, mode='lines', name='BSS (Discharge)', stackgroup='positive'), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.datetime, y=res.P_discharge_ev, mode='lines', name='EV (Discharge)', stackgroup='positive'), row=1, col=1)

    # Add traces for negative power values (consumption)
    fig.add_trace(go.Scatter(x=res.datetime, y=-res.P_load, mode='lines', name='Load', stackgroup='negative'), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.datetime, y=-res.P_charge_bss, mode='lines', name='BSS (Charge)', stackgroup='negative'), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.datetime, y=-res.P_charge_ev, mode='lines', name='EV (Charge)', stackgroup='negative'), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.datetime, y=-res.P_exp, mode='lines', name='Exports', stackgroup='negative'), row=1, col=1)

    # Second subplot for SOC values
    fig.add_trace(go.Scatter(x=res.datetime, y=res.SOC_ev / res.C_ev * 100, mode='lines', name='EV SOC [%]'), row=2, col=1)
    fig.add_trace(go.Scatter(x=res.datetime, y=res.SOC_bss / res.C_bss * 100, mode='lines', name='Battery SOC [%]'), row=2, col=1)

    if vert_lines: # Add vertical lines for EV arrival and departure times
        for t in res.t_arr:
            fig.add_vline(x=res.datetime[t], line=dict(color="green", width=1, dash="dash"), row=2, col=1)
        for t in res.t_dep:
            fig.add_vline(x=res.datetime[t], line=dict(color="red", width=1, dash="dash"), row=2, col=1)

    # Update layout for the figure
    fig.update_layout(
        title="Simulation Results",
        xaxis_title="Time",
        yaxis_title="Power [kW]",
        yaxis2_title="State of Charge [%]",
        legend_title="Legend",
        template="plotly_white",
        hovermode="x unified"
    )

    # Show the interactive plot
    fig.show()
    return
