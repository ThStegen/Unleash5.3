import numpy as np
import pandas as pd
from datetime import timedelta, datetime

from param import delta_t, SOC_target_ev, C_ev
from dataclasses import dataclass, field
from typing import Any, Optional

class SizingParam:
    def __init__(self, Params, Grid, PV, BSS, EV, HP, WB, Gen, C_grid=None, 
                 C_pv=None, P_nom_PV=None, 
                 C_bss=None, P_nom_BSS=None, 
                 C_ev=None, P_nom_EV=None, 
                 C_env_HP=None, C_air_HP=None, P_nom_HP=None, COP=None, 
                 C_water_WB=None, P_nom_WB=None, 
                 P_nom_gen=None):
        if Params.Grid:
            self.Grid = Grid
            self.C_grid = C_grid if not Grid else None
            if not Grid and C_grid is None:
                raise ValueError("C_grid must be provided if Grid sizing is not done.")
        if Params.PV:
            self.PV = PV
            self.C_pv = C_pv if not PV else None
            if not PV and C_pv is None:
                raise ValueError("C_pv must be provided if PV sizing is not done.")
            self.P_nom_PV = P_nom_PV if not PV else None
            if not PV and P_nom_PV is None:
                raise ValueError("P_nom_PV must be provided if PV sizing is not done.")
        if Params.BSS:
            self.BSS = BSS
            self.C_bss = C_bss if not BSS else None
            if not BSS and C_bss is None:
                raise ValueError("C_bss must be provided if BSS sizing is not done.")
            self.P_nom_BSS = P_nom_BSS if not BSS else None
            if not BSS and P_nom_BSS is None:
                raise ValueError("P_nom_BSS must be provided if BSS sizing is not done.")
        if Params.EV:
            self.EV = EV
            self.C_ev = C_ev if not EV else None
            if not EV and C_ev is None:
                raise ValueError("C_ev must be provided if EV sizing is not done.")
            self.P_nom_EV = P_nom_EV if not EV else None
            if not EV and P_nom_EV is None:
                raise ValueError("P_nom_EV must be provided if EV sizing is not done.")
        if Params.HP:
            self.HP = HP
            self.C_env_HP = C_env_HP if not HP else None
            if not HP and C_env_HP is None:
                raise ValueError("C_env_HP must be provided if HP sizing is not done.")
            self.C_air_HP = C_air_HP if not HP else None
            if not HP and C_air_HP is None:
                raise ValueError("C_air_HP must be provided if HP sizing is not done.")
            self.P_nom_HP = P_nom_HP if not HP else None
            if not HP and P_nom_HP is None:
                raise ValueError("P_nom_HP must be provided if HP sizing is not done.")
            self.COP = COP if not HP else None
            if not HP and COP is None:
                raise ValueError("COP must be provided if HP sizing is not done.")
        if Params.WB:
            self.WB = WB
            self.C_water_WB = C_water_WB if not WB else None
            self.P_nom_WB = P_nom_WB if not WB else None

        if Params.Gen:
            self.Gen = Gen
            self.P_nom_gen = P_nom_gen if not Gen else None
            if not Gen and P_nom_gen is None:
                raise ValueError("P_nom_gen must be provided if Gen sizing is not done.")
        
class Parameters():
    def __init__(self, Grid, PV, BSS, EV, HP, WB, Flex, Gen):
        self.Grid = Grid   
        self.PV = PV
        self.BSS = BSS
        self.EV = EV
        self.HP = HP
        self.WB = WB
        self.Flex = Flex
        self.Gen = Gen


    @dataclass
    class WB:
        type: int
        P_max: float
        T_set: float
        specific_heat: float
        Volume: float
        C_th: float
        P_ref: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        P_use: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        P_loss: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        T_ref: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        t_use: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=bool))
        alpha: float = 1.0

        def __post_init__(self):
            # Ensure numpy arrays and correct dtypes
            self.P_ref = np.asarray(self.P_ref, dtype=float)
            self.P_use = np.asarray(self.P_use, dtype=float)
            self.P_loss = np.asarray(self.P_loss, dtype=float)
            self.T_ref = np.asarray(self.T_ref, dtype=float)
            self.t_use = np.asarray(self.t_use, dtype=bool)


    @dataclass
    class HP:
        type: int
        P_max: float
        COP: float
        C_th: float
        A_u: float
        P_ref: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        T_ref: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        T_set: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        T_wall: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        P_loss: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        T_out: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
        alpha: float = 0.0  # Alpha_HP

        def __post_init__(self):
            self.P_ref = np.asarray(self.P_ref, dtype=float)
            self.T_ref = np.asarray(self.T_ref, dtype=float)
            self.T_set = np.asarray(self.T_set, dtype=float)
            self.T_wall = np.asarray(self.T_wall, dtype=float)
            self.P_loss = np.asarray(self.P_loss, dtype=float)
            self.T_out = np.asarray(self.T_out, dtype=float)


    @dataclass
    class EV:
        P_ref: Optional[np.ndarray] = field(default_factory=lambda: np.array([[]]))  # 2D
        a_EV: Optional[np.ndarray] = field(default_factory=lambda: np.array([[]], dtype=bool))  # BitMatrix
        t_arr: Optional[np.ndarray] = field(default_factory=lambda: np.array([[]], dtype=bool))
        t_dep: Optional[np.ndarray] = field(default_factory=lambda: np.array([[]], dtype=bool))
        SoC_target: float = 0.0
        SoC_ref: Optional[np.ndarray] = field(default_factory=lambda: np.array([[]]))
        SoC_arr: Optional[np.ndarray] = field(default_factory=lambda: np.array([[]]))
        SoC_init: float = 0.0
        eta: float = 1.0
        Cap: float = 0.0
        P_max: float = 0.0
        alpha: float = 0.0

        def __post_init__(self):
        self.P_ref = np.asarray(self.P_ref, dtype=float)
        self.a_EV = np.asarray(self.a_EV, dtype=bool)
        self.t_arr = np.asarray(self.t_arr, dtype=bool)
        self.t_dep = np.asarray(self.t_dep, dtype=bool)
        self.SoC_ref = np.asarray(self.SoC_ref, dtype=float)
        self.SoC_arr = np.asarray(self.SoC_arr, dtype=float)


class Results:
    def __init__(self, filename, start_time, n_days, yearly_kwh, yearly_km):
        self.start_time = start_time 
        self.t_s = int(n_days*24/delta_t)                # Total number of discrete time steps in the simulation
        self.n_days = n_days
        self.yearly_kwh = yearly_kwh
        self.yearly_km = yearly_km
        self.t = np.arange(0,self.t_s)

        # Initialize SOCs
        self.SOC_bss_i = 0.5  

        # Load data from CSV files into pandas DataFrames
        self.df = pd.read_csv(filename, delimiter=';', index_col="DateTime", parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')#, date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        self.P_load = np.array([self.df.loc[self.start_time + timedelta(hours=t*delta_t)]["Load"].clip(min=0) * self.yearly_kwh for t in self.t])
        self.P_pv_max = np.array([self.df.loc[self.start_time + timedelta(hours=t*delta_t)]["PV"].clip(min=0) for t in self.t])
        self.EV_connected = np.array([self.df.loc[self.start_time + timedelta(hours=t*delta_t)]["EV"] for t in self.t])
        self.datetime = [self.start_time + timedelta(hours=t*delta_t) for t in self.t]


        self.SOC_ev_i = [SOC_target_ev*C_ev - (self.EV_connected[t]*self.yearly_km) / (5e6) for t in range(self.t_s) if self.EV_connected[t] > 0 and (t == 0 or self.EV_connected[t-1] == 0)]
        self.t_arr = [t for t in range(self.t_s) if self.EV_connected[t] > 0 and (t == 0 or self.EV_connected[t-1] == 0)]
        self.t_dep = [t for t in range(self.t_s) if self.EV_connected[t] == 0 and (t > 0 and self.EV_connected[t-1] > 0)] + ([self.t_s-1] if self.EV_connected[-1] > 0 else [])
        self.EV_connected = [1 if self.EV_connected[t] > 0 else 0 for t in range(self.t_s)]

    def save_sizing_results(self, m):
        self.C_bss = m.C_bss.value
        self.P_nom_BSS = m.P_nom_BSS.value
        self.C_pv = m.C_pv.value
        self.P_nom_PV = m.P_nom_PV.value
        self.C_ev = m.C_ev.value
        self.P_nom_EV = m.P_nom_EV.value
        self.P_nom_gen = m.P_nom_gen.value

    def save_results(self, m):
        self.P_imp = np.array([m.P_imp[t].value for t in m.periods])
        self.P_exp = np.array([m.P_exp[t].value for t in m.periods])
        self.P_pv = np.array([m.P_pv[t].value for t in m.periods])
        self.P_charge_bss = np.array([m.P_charge_bss[t].value for t in m.periods])
        self.P_discharge_bss = np.array([m.P_discharge_bss[t].value for t in m.periods])
        self.P_charge_ev = np.array([m.P_charge_ev[t].value for t in m.periods])
        self.P_discharge_ev = np.array([m.P_discharge_ev[t].value for t in m.periods])
        self.P_gen = np.array([m.P_gen[t].value for t in m.periods])
        self.SOC_ev = np.array([m.SOC_ev[t].value for t in m.periods])
        self.SOC_bss = np.array([m.SOC_bss[t].value for t in m.periods])
        self.objective = m.objective()
        return



class ResultsFlex:
    def __init__(self, user, fix_df, flex_df, pv, start_time, n_days):
        self.start_time = start_time 
        self.t_s = int(n_days*24/delta_t)                # Total number of discrete time steps in the simulation
        self.timestep = delta_t
        self.n_days = n_days
        self.t = np.arange(0,self.t_s)

        # Initialize SOCs
        self.SOC_bss_i = 0.5  

        # # Load data from CSV files into pandas DataFrames
        self.P_load = np.array([fix_df.loc[self.start_time + timedelta(minutes=int(t*delta_t*60))]["BaseLoad"].clip(min=0) for t in self.t])
        self.P_pv_max = np.array([pv.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["S"].clip(min=0) for t in self.t])
        #self.P_pv_max = np.array([pv.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))][user["PV_data"]["Azimut"]].clip(min=0) for t in self.t])
        self.datetime = [self.start_time + timedelta(minutes=int(60*t*delta_t)) for t in self.t]


        # EV profile
        self.P_EV = np.array([fix_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["P_EV"] for t in self.t])

        self.EV_plugged = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["EV_plugged"] for t in self.t])
        self.EV_arrival = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["EV_arrival"] for t in self.t])
        self.EV_departure = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["EV_departure"] for t in self.t])
        self.SoC_ref_EV = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["SoC_ref_EV"] for t in self.t])
        self.SoC_arr_EV = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["SoC_arr_EV"] for t in self.t])

        # HP profile
        self.P_HP = np.array([fix_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["P_HP"].clip(min=0) for t in self.t])

        self.T_set_HP = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["T_set_HP"] for t in self.t])
        self.T_ref_HP = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["T_ref_HP"] for t in self.t])
        self.T_wall_HP = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["T_wall_HP"] for t in self.t])
        self.T_out_HP = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["T_out_HP"] for t in self.t])
        self.P_loss_HP = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["P_loss_HP"] for t in self.t])
        
        # WB profile
        self.P_WB = np.array([fix_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["P_WB"].clip(min=0) for t in self.t])

        self.P_loss_WB = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["P_loss_WB"] for t in self.t])
        self.P_use_WB = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["P_use_WB"] for t in self.t])
        self.T_ref_WB = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["T_ref_WB"] for t in self.t])
        self.T_set_WB = np.array([flex_df.loc[self.start_time + timedelta(minutes=int(60*t*delta_t))]["T_set_WB"] for t in self.t])



    def save_sizing_results(self, m):
        self.C_bss = m.C_bss.value
        self.P_nom_BSS = m.P_nom_BSS.value
        self.C_pv = m.C_pv.value
        self.P_nom_PV = m.P_nom_PV.value
        self.C_ev = m.C_ev.value
        self.P_nom_EV = m.P_nom_EV.value
        self.P_nom_gen = m.P_nom_gen.value

    def save_results(self, m):
        self.P_imp = np.array([m.P_imp[t].value for t in m.periods])
        self.P_exp = np.array([m.P_exp[t].value for t in m.periods])
        self.P_pv = np.array([m.P_pv[t].value for t in m.periods])
        self.P_charge_bss = np.array([m.P_charge_bss[t].value for t in m.periods])
        self.P_discharge_bss = np.array([m.P_discharge_bss[t].value for t in m.periods])
        self.P_charge_ev = np.array([m.P_charge_ev[t].value for t in m.periods])
        self.P_discharge_ev = np.array([m.P_discharge_ev[t].value for t in m.periods])
        self.P_gen = np.array([m.P_gen[t].value for t in m.periods])
        self.SOC_ev = np.array([m.SOC_ev[t].value for t in m.periods])
        self.SOC_bss = np.array([m.SOC_bss[t].value for t in m.periods])
        self.objective = m.objective()
        return