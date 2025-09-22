# from datetime import datetime
# import utils, models
# from classes import SizingParam, Parameters, ResultsFlex
# import os
# import json
# import pandas as pd
# import numpy as np


def import_from_resflex(folder_path):
    case_path = os.path.join("input", folder_path, "case.json")
    with open(case_path, "r") as f: case = json.load(f)
    
    users_path = os.path.join("input", folder_path, "Param.json")
    with open(users_path, "r") as f: users_json = json.load(f)

    users = [dict(u) for u in list(users_json.values())]
    SizingParams = [SizingParam(Params, False, False, False, False, False, False, False, C_grid=64*230e-3, 
                 C_pv=u["PV_data"]["Pmax"], P_nom_PV=u["PV_data"]["Pmax"], 
                 C_bss=u["BSS_data"]["Capacity"], P_nom_BSS=u["BSS_data"]["Pmax"], 
                 C_ev=u["EV_data"]["Capacity"], P_nom_EV=u["EV_data"]["Pmax"], 
                 C_env_HP=u["HP_data"]["C_env"], C_air_HP=u["HP_data"]["C_air"], P_nom_HP=u["HP_data"]["P_nom"], COP=u["HP_data"]["COP"], 
                 C_water_WB=u["WB_data"]["Volume"]*u["WB_data"]["specific_heat"], P_nom_WB=u["WB_data"]["Pmax"], 
                 P_nom_gen=None) for u in users]
    return SizingParams, case



def run(model, data):
    model, data = utils.solve_model(model, data)
    if model and data:
        utils.print_sizing_results(data)
        #utils.check_res(data)
        utils.print_res(data)
        utils.plot_res2(data, vert_lines=False)
    return data
"""
input_folder = "Test2"                              # Folder containing the input files
start_time = datetime(2024, 1, 1, 0, 0, 0)          # Starting date for the simulation
n_days = 1                                          # Number of days to simulate

Params = Parameters(Grid = True, 
                   PV = True, 
                   BSS = True, 
                   EV = True, 
                   HP = True, 
                   WB = True,
                   Flex = False, 
                   Gen = False)
users, case = import_from_resflex(input_folder)

dfs_fix = [pd.read_csv(os.path.join("input", input_folder, f"household_{u+1}_ref.csv"), index_col=0, parse_dates=True) for u in range(len(users))]
dfs_flex = [pd.read_csv(os.path.join("input", input_folder, f"household_{u+1}_flex.csv"), index_col=0, parse_dates=True) for u in range(len(users))]


pvs = pd.read_csv(os.path.join("input", "pv.csv"), index_col=0, parse_dates=True)
timestep_min = 24*60 / case["n_periods"]
tstep = int(timestep_min)  # minutes per timestep
pvs = pvs.resample(f'{tstep}min').mean()

data = [ResultsFlex(users[i], dfs_fix[i], dfs_flex[i], pvs, start_time, n_days) for i in range(len(users))]

result_fix = [None] * len(users)
result_flex = [None] * len(users)
for u in range(len(users)):
    model = models.create_model(Params, users[u], data[u])
    result_fix[u] = run(model, data)
    Params.Flex = True
    model = models.create_model(Params, users[u], data[u])
    result_flex[u] = run(model, data)
    Params.Flex = False



def run_post_pf(data):
    for d in data:
        d.save_sizing_results(d.m)
        utils.save_results(d, d.m)
        utils.check_res(d)
        utils.print_res(d)
        utils.plot_res2(d, vert_lines=False)
    return data


#pf_fix = run_post_pf(data_fix, case)
#pf_flex = run_post_pf(data_flex, case)
"""

def cotisations_sociales(brut):
    """Calcule les cotisations sociales (13,07)."""
    return brut * 0.1307

def revenu_imposable(brut):
    """Passe du brut au revenu imposable apres cotisations sociales."""
    return brut - cotisations_sociales(brut)

def impot_progressif(ri_annuel):
    """
    Calcule l'impot des personnes physiques (IPP) selon bareme progressif 2025.
    Bareme isole, sans enfants.
    """
    # Tranches annuelles 2025 Belgique (approx)
    tranches = [
        (15340, 0.25),
        (27090, 0.40),
        (48430, 0.45),
        (float("inf"), 0.50)
    ]
    
    # Quotite exemptee
    qf = 10570
    ri_annuel -= qf
    if ri_annuel < 0:
        return 0
    
    imp = 0
    prev = 0
    for plafond, taux in tranches:
        if ri_annuel > plafond:
            imp += (plafond - prev) * taux
            prev = plafond
        else:
            imp += (ri_annuel - prev) * taux
            break
    return imp

def net_mensuel(brut_mensuel):
    """
    Calcule le net mensuel a partir du brut mensuel.
    """
    # Revenu annuel imposable
    ri_annuel = revenu_imposable(brut_mensuel * 12)
    imp_annuel = impot_progressif(ri_annuel)
    
    # Net annuel = brut - cotisations - impot
    net_annuel = brut_mensuel * 12 - cotisations_sociales(brut_mensuel * 12) - imp_annuel
    return net_annuel / 12

# --- Simulation de scenarios ---
brut_tp = 4456   # Temps plein brut
brut_1_5 = 796   # 1/5 brut
brut_2_5 = 1592  # 2/5 brut

scenarios = {
    "Temps plein": brut_tp,
    "TP + 1/5": brut_tp + brut_1_5,
    "TP + 2/5": brut_tp + brut_2_5,
    "4/5 + 1/5": 0.8 * brut_tp + brut_1_5,
    "4/5 + 2/5": 0.8 * brut_tp + brut_2_5,
    "3/5 + 2/5": 0.6 * brut_tp + brut_2_5,
    "3/5 + 1/5": 0.6 * brut_tp + brut_1_5,
    "1/2 + 2/5": 0.5 * brut_tp + brut_2_5,
    "1/2 + 1/5": 0.5 * brut_tp + brut_1_5,
}

# Affichage des resultats
for nom, brut in scenarios.items():
    print(f"{nom:12s} | Brut: {brut:7.2f} ? | Net: {net_mensuel(brut):7.2f} ?")