from param import delta_t, inv_hor, C_pv_max, C_bss_max, C_ev_max, C_inv_max
from param import SOC_target_ev, SOC_min_bss, SOC_max_bss, SOC_i_bss, SOC_min_ev, SOC_max_ev
from param import eff_bss, eff_ev
from param import PI_gen, PI_imp, PI_exp, PI_bss, PI_ev, PI_c_inv, PI_c_bss, PI_c_pv, PI_c_gen
from pyomo.environ import ConcreteModel, Param, Var, Objective, Constraint, NonNegativeReals, minimize
import utils

def create_model(Params, SizingParams, data):
    model = base_model(data)
    if Params.Grid: model = add_grid(model, SizingParams, data)
    if Params.PV: model = add_PV(model, SizingParams, data)
    if Params.BSS: model = add_BSS(model, SizingParams, data)
    if Params.EV: model = add_EV(model, SizingParams, data)
    if Params.HP: model = add_HP(model, SizingParams, data)
    if Params.Flex: model = add_flex(model, SizingParams, data)
    if Params.Gen: model = add_gen(model, SizingParams, data)
    model = power_balance(model, Params)
    model = objective(model, Params, data)
    return model

def base_model(data):
    m = ConcreteModel()
    m.periods = data.t
    m.timestep = data.timestep
    m.P_load = Var(m.periods, within=NonNegativeReals)  # Load power
    m.P_fix_cstr = Constraint(m.periods, rule=lambda m, t: m.P_load[t] == data.P_load[t])  # Fixed load constraint
    return m

def add_grid(m, sizing, data):
    m.P_imp = Var(m.periods, within=NonNegativeReals)  # Power imported from the grid
    m.P_exp = Var(m.periods, within=NonNegativeReals)  # Power exported to the grid

    if sizing.Grid:
        m.P_max_grid = Var(within=NonNegativeReals)  # Battery inverter nominal power
    else:
        m.P_max_grid = Param(initialize=sizing.C_grid)  # Battery inverter nominal power


    # Grid power constraints
    m.grid_import_cstr = Constraint(m.periods, rule=lambda m, t: m.P_imp[t] <= m.P_max_grid)
    m.grid_export_cstr = Constraint(m.periods, rule=lambda m, t: m.P_exp[t] <= m.P_max_grid)

    return m

def add_PV(m, sizing, data):
    m.P_pv = Var(m.periods, within=NonNegativeReals)  # PV power output

    if sizing.PV: 
        m.P_nom_pv = Var(within=[0, C_inv_max])  # Nominal power for PV inverter
        m.C_pv = Var(within=[0, C_pv_max])  # PV system size
    else:
        m.P_nom_pv = Param(initialize=sizing.P_nom_PV)
        m.C_pv = Param(initialize=sizing.C_pv)  # PV system size

    # PV power constraints
    m.PV_max_cstr = Constraint(m.periods, rule=lambda m, t: m.P_pv[t] <= data.P_pv_max[t] * m.C_pv)

    return m

def add_BSS(m, sizing, data):
    m.P_charge_bss = Var(m.periods, within=NonNegativeReals)    # Battery charging power
    m.P_discharge_bss = Var(m.periods, within=NonNegativeReals) # Battery discharging power
    m.SOC_bss = Var(m.periods, within=NonNegativeReals)         # Battery state of charge

    if sizing.BSS: 
        m.P_nom_bss = Var(within=[0, C_inv_max])  # Nominal power for PV inverter
        m.C_bss = Var(within=[0, C_bss_max])  # PV system size
    else:
        m.P_nom_bss = Param(initialize=sizing.P_nom_BSS)
        m.C_bss = Param(initialize=sizing.C_bss)  # PV system size
    
    # Battery constraints
    m.bss_charge_cstr = Constraint(m.periods, rule=lambda m, t: m.P_charge_bss[t] <= m.P_nom_bss)
    m.bss_discharge_cstr = Constraint(m.periods, rule=lambda m, t: m.P_discharge_bss[t] <= m.P_nom_bss)
    m.bss_soc_cstr = Constraint(m.periods, rule=lambda m, t: m.SOC_bss[t] <= m.C_bss)
    m.soc_max_bss_cstr = Constraint(m.periods, rule=lambda m, t: m.SOC_bss[t] <= SOC_max_bss * m.C_bss)
    m.soc_min_bss_cstr = Constraint(m.periods, rule=lambda m, t: m.SOC_bss[t] >= SOC_min_bss * m.C_bss)

    # Battery Energy constraints
    m.SOC_bss_cstr = Constraint(m.periods, expr=lambda m,t: m.SOC_bss[t] == SOC_i_bss*m.C_bss + delta_t*(m.P_charge_bss[t]*eff_bss-m.P_discharge_bss[t]) if t == 0
                                    else m.SOC_bss[t] == m.SOC_bss[t-1] + delta_t*(m.P_charge_bss[t]*eff_bss-m.P_discharge_bss[t]))
    return m

def add_EV(m, sizing, data):
    m.P_charge_ev = Var(m.periods, within=NonNegativeReals)  # EV charging power
    m.P_discharge_ev = Var(m.periods, within=NonNegativeReals)  # EV discharging power
    m.SOC_ev = Var(m.periods, within=NonNegativeReals)  # EV state of charge
    
    if sizing.EV: 
        m.P_nom_ev = Var(within=[0, C_inv_max])  # Nominal power for PV inverter
        m.C_ev = Var(within=[0, C_ev_max])  # PV system size
    else:
        m.P_nom_ev = Param(initialize=sizing.P_nom_EV)
        m.C_ev = Param(initialize=sizing.C_ev)  # PV system size

    # EV constraints
    m.ev_charge_cstr = Constraint(m.periods, rule=lambda m, t: m.P_charge_ev[t] <= m.P_nom_ev * data.EV_plugged[t])
    m.ev_discharge_cstr = Constraint(m.periods, rule=lambda m, t: m.P_discharge_ev[t] <= m.P_nom_ev * data.EV_plugged[t])
    m.ev_soc_cstr = Constraint(m.periods, rule=lambda m, t: m.SOC_ev[t] <= m.C_ev)
    m.soc_max_ev_cstr = Constraint(m.periods, rule=lambda m, t: m.SOC_ev[t] <= SOC_max_ev * m.C_ev if t in data.EV_plugged else Constraint.Skip)
    m.soc_min_ev_cstr = Constraint(m.periods, rule=lambda m, t: m.SOC_ev[t] >= SOC_min_ev * m.C_ev if t in data.EV_plugged else Constraint.Skip)
    #m.SOC_ev_cstr = Constraint(m.periods, expr=lambda m,t: m.SOC_ev[t] == data.SOC_ev_i[data.t_arr.index(t)] + delta_t*(m.P_charge_ev[t]*eff_ev-m.P_discharge_ev[t]) if t in data.t_arr
    #                               else m.SOC_ev[t] == m.SOC_ev[t-1] + delta_t*(m.P_charge_ev[t]*eff_ev-m.P_discharge_ev[t]) if t in data.EV_plugged else m.SOC_ev[t] == 0)
    
    m.SOC_ev_cstr = Constraint(m.periods, expr=lambda m,t: m.SOC_ev[t] == data.SOC_ev_i[data.t_arr.index(t)] + delta_t*(m.P_charge_ev[t]*eff_ev-m.P_discharge_ev[t]) if t in data.t_arr
                                   else m.SOC_ev[t] == m.SOC_ev[t-1] + delta_t*(m.P_charge_ev[t]*eff_ev-m.P_discharge_ev[t]) if t >0 else m.SOC_ev[t] == 0.5*m.C_ev)
    # EV target SOC constraint
    m.EV_target_cstr = Constraint(m.connections, expr=lambda m, t: m.SOC_ev[t] >= SOC_target_ev*data.EV_dep[t])

    return m

def add_flex(m, data):
    m.P_flex = Var(m.periods, within=NonNegativeReals)  # Flexible load power

    # Flexible load constraints
    m.flex_cstr = Constraint(m.periods, rule=lambda m, t: m.P_flex[t] <= data.P_flex_max)

    return m

def add_fix(m, data):
    m.P_flex = Var(m.periods, within=NonNegativeReals)  # Fixed load power
    # Fixed load constraints
    m.fix_cstr = Constraint(m.periods, rule=lambda m, t: m.P_flex[t] == data.P_EV[t] + data.P_WB[t] + data.P_HP[t])

def add_HP(m, sp, data):
    m.P_HP = Var(m.periods, within=NonNegativeReals)  # Heat pump power
    m.T_HP = Var(m.periods, within=NonNegativeReals)  # Indoor temperature
    m.J_HP = Var(m.periods, within=NonNegativeReals)  # HP discomfort

    m.HP_pmax = Constraint(m.periods, rule=lambda m, p: m.P_HP[p] <= sp.P_max_HP)
    m.HP_temp = Constraint(m.periods, rule=lambda m,p: m.T_HP[p] == m.timestep * sp.C_HP * (sp.COP_HP *m.P_HP[p] - data.P_loss_HP[p]) 
                                                                    + (data.T_set_HP[0] if p == 0 else m.T_HP[p-1]))
    m.HP_energy = Constraint(expr=sum(m.P_HP[p] for p in m.periods) == sum(data.P_HP(p) for p in m.periods))
    m.HP_disc = Constraint(m.periods, rule=lambda m,p: m.J_HP >= sp.alpha_HP * (min(data.T_set_HP[p], data.T_ref_HP[p])- m.T_HP[p]))
    return m

def add_EV(m, sp, data):
    m.P_EV = Var(m.periods, within=NonNegativeReals)  # Electric vehicle power
    m.s_EV = Var(m.periods, within=NonNegativeReals)  # EV SoC
    m.J_EV = Var(m.periods, within=NonNegativeReals)  # EV discomfort

    m.EV_pmax = Constraint(m.periods, rule=lambda m, p: m.P_EV[p] <= sp.P_nom_EV*data.EV_plugged[p])
    m.EV_soc = Constraint(m.periods, rule=lambda m,p: m.s_EV[p] == m.timestep * sp.C_ev * (m.P_EV[p]) 
                                                                        + (data.SoC_ref_EV[0] if p == 0 else m.s_EV[p-1]))
    
    m.EV_energy = Constraint(expr=sum(m.P_EV[p] for p in m.periods) == sum(data.P_EV[p] for p in m.periods))
    m.EV_disc = Constraint(m.periods, rule=lambda m,p: m.J_EV >= sp.alpha_EV * (data.SOC_ref_EV[p] - m.s_EV[p]))
    
    return m

def add_gen(m, SizingParams, data):
    if SizingParams.P_max_gen is None:
        m.P_max_gen = Var(initialize=0, within=NonNegativeReals)                 # Maximum generator power
    else:
        m.P_max_gen = Param(initialize=SizingParams.P_max_gen)  # Maximum generator power
    m.P_gen = Var(m.periods, within=NonNegativeReals)
    # Generator maximum and minimum power constraints:
    m.gen_max_cstr = Constraint(m.periods, expr=lambda m, t: m.P_gen[t] <= m.P_max_gen)
    return m

def power_balance(m, Params):
    # Power balance constraint
    m.power_balance_cstr = Constraint(
        m.periods,
        rule=lambda m, t: (
            (m.P_imp[t] if Params.Grid else 0) +
            (m.P_pv[t] if Params.PV else 0) +
            (m.P_gen[t] if Params.Gen else 0) +
            (m.P_discharge_bss[t] if Params.BSS else 0) +
            (m.P_discharge_ev[t] if Params.EV else 0)
            ==
            (m.P_exp[t] if Params.Grid else 0) +
            (m.P_charge_bss[t] if Params.BSS else 0) +
            (m.P_charge_ev[t] if Params.EV else 0) +
            (m.P_flex[t] if Params.Flex else 0) +
            m.P_load[t]
        )
    )
    return m

def objective(m, Params, data):
    # Define the objective function
    m.objective = Objective(
        sense=minimize,
        expr=(
            365 * (delta_t * sum(
                (m.P_gen[t] * PI_gen if Params.Gen else 0) +
                (m.P_imp[t] * PI_imp if Params.Grid else 0) -
                (m.P_exp[t] * PI_exp if Params.Grid else 0)
                for t in m.periods) +
                sum(m.P_discharge_bss[t]**2 * PI_bss if Params.BSS else 0 for t in m.periods) +
                sum(m.P_charge_bss[t]**2 * PI_bss if Params.BSS else 0 for t in m.periods) +
                sum(m.P_discharge_ev[t]**2 * PI_ev if Params.BSS else 0 for t in m.periods) +
                sum(m.P_charge_ev[t]**2 * PI_ev if Params.BSS else 0 for t in m.periods)
               
            ) / data.n_days
            + (
                (PI_c_pv * m.C_pv  + PI_c_inv * m.P_nom_pv if Params.Sizing.PV == True else 0) +
                (PI_c_bss * m.C_bss + PI_c_inv * m.P_nom_bss if Params.Sizing.BSS == True else 0) +
                (PI_c_gen * m.P_max_gen if Params.Sizing.Gen == True else 0) 
            ) / inv_hor

        )
    )
    return m
