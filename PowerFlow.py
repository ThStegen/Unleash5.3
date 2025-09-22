import os
import sys
import argparse
import pandas as pd
import numpy as np

import pandapower as pp
import pandapower.networks as pn

n_steps = 96
BASE_KV = 0.4

def clear_power_elements(net):
    """Remove existing loads/sgens/gens from the network to prepare for a new time-step."""
    for tbl in ("load", "sgen", "gen"):
        if getattr(net, tbl) is not None:
            net.__setattr__(tbl, net.__getattribute__(tbl).iloc[0:0].copy())

def apply_profile_to_net(net, profile_row_mw):
    """
    profile_row_mw : 1D array-like length 122 with values in MW (positive injection, negative consumption)
    - For >0 we create sgen (distributed generation / injection)
    - For <0 we create loads
    We'll create one element per bus. We keep names for traceability.
    """
    n_buses = len(profile_row_mw)
    # safety: ensure we have at least n_buses buses in net
    if len(net.bus) < n_buses:
        raise ValueError(f"Network has {len(net.bus)} buses but profile has {n_buses} columns")

    # Remove prior elements
    clear_power_elements(net)

    for i, p in enumerate(profile_row_mw):
        bus_idx = i  # we used bus order 0..121
        if np.isnan(p) or abs(p) < 1e-12:
            # do nothing for zero
            continue
        if p < 0:
            # consumption: create load with p_mw = -p
            pp.create_load(net, bus=bus_idx, p_mw=float(-p), q_mvar=0.0, name=f"Load_bus_{bus_idx}")
        else:
            # injection: create sgen (distributed generator/injection)
            # sgen is a static generator (p positive injects into bus).
            pp.create_sgen(net, bus=bus_idx, p_mw=float(p), q_mvar=0.0, name=f"SGen_bus_{bus_idx}")



def replace_trafos(net):
    index = list(net.trafo.index)
    from_bus = list(net.trafo.hv_bus)
    to_bus = list(net.trafo.lv_bus)
    for idx in index:
        pp.drop_trafos(net, [idx])
        net.bus.drop(from_bus, inplace=True)
        net.ext_grid.drop(from_bus, inplace=True)
        pp.create_ext_grid(net, to_bus[0], vm_pu=1.0, va_degree=0.0)
        if not net.bus_geodata.empty:
            net.bus_geodata.drop(from_bus, inplace=True)
        pp.create_continuous_elements_index(net, start=0)

    return net


def clean_lines(net):
    for idx in net.line.index:
        net.line.c_nf_per_km[idx] = 0
        net.line.max_i_ka[idx] = 1000
        # Maybe remove i_max
    return net


def single_volt_lvl(net, voltage):
    for idx in net.bus.index:
        net.bus.v_kv = voltage
    return net


def add_PV(net):
    # init active power
    p_mw = np.zeros(len(net.bus))
    # set rated sizes
    sn_mva = [10 for i in len(net.bus)]

    # create sgen
    if len(net.sgen.name):
        net.sgen.drop(net.sgen.index, inplace=True)
    for idx, bus in enumerate(net.bus):
        pp.create_sgen(net, bus, p_mw=p_mw[idx], q_mvar=0, sn_mva=sn_mva[idx], name=name[idx],
                       index=None, scaling=1.0, type='wye', in_service=True, controllable=True)

    # add sgen at slack bus
    pp.create_sgen(net, 0, p_mw=0, q_mvar=0, sn_mva=1, name='Slack Gen',
                   index=None, scaling=1.0, type='wye', in_service=True, controllable=True)

    return net


def create_net():
    
    net = pn.create_dickert_lv_network(
        feeders_range='short', linetype='cable', customer='multiple', case='average')
    net = replace_trafos(net)
    net = clean_lines(net)
    net = single_volt_lvl(net, BASE_KV)
    net = add_PV(net)
    return net

def run_timeseries(csv_path, outdir='results/powerflow'):
    # Read CSV
    df = pd.read_csv(csv_path, header=None)
    
    # convert to numpy array in MW
    data_mw = df.values.astype(float) /1000

    # Create network
    net = create_net()

    # Ensure output dir
    os.makedirs(outdir, exist_ok=True)

    # We'll store mean voltage and number of converged / not converged
    summary_cols = ["step", "converged", "mean_vm_pu", "min_vm_pu", "max_vm_pu"]
    summary = []

    # For each time step:
    for t in range(n_steps):
        row = data_mw[t, :]  # length 122
        print(f"Step {t+1}/{n_steps} : applying profile and running powerflow...", end=' ')
        # Apply profile
        apply_profile_to_net(net, row)

        # Run power flow
        try:
            pp.runpp(net, algorithm='nr')  # Newton-Raphson (can change)
            converged = True
        except Exception as e:
            print(f"\nPower flow did not converge at step {t+1}: {e}")
            converged = False

        # Collect results
        vm_pu = net.res_bus.vm_pu if converged else pd.Series([np.nan]*len(net.bus), index=net.bus.index)
        p_bus_inj = None
        # compute injections per bus from sgen/load and ext_grid
        # net.res_bus has p_mw from bus injection only if you build accordingly; simpler: use net.res_bus.p_mw if present
        # We'll save vm and also the elements tables for traceability.
        # Save bus voltages
        res_bus_df = net.res_bus[["vm_pu"]].copy()

        # Summarize
        mean_vm = float(np.nanmean(res_bus_df["vm_pu"].values)) if converged else np.nan
        min_vm = float(np.nanmin(res_bus_df["vm_pu"].values)) if converged else np.nan
        max_vm = float(np.nanmax(res_bus_df["vm_pu"].values)) if converged else np.nan
        summary.append({
            "step": t+1,
            "converged": converged,
            "mean_vm_pu": mean_vm,
            "min_vm_pu": min_vm,
            "max_vm_pu": max_vm
        })
        print("done." if converged else "failed.")

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(outdir, "results_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"All done. Summary saved to {summary_csv}")
    
if __name__ == "__main__":
    run_timeseries('results/implicit/capacity.csv')