"""
Tests of step 2, part A: plans in the ExecutionResult (FrenetiX). The new path (executor's plan loader ->
ExecutionResult.load_plans -> FrenetixScenarioEvaluator._plans_table) must give the same table as the old
_planned_trajectories, which read the agents' logs.csv directly.
Run from the project root: python3 tests/test_step2_plans.py
"""
import os, sys, pickle, tempfile, functools
import numpy as np
import pandas as pd
from types import SimpleNamespace
sys.path.insert(0, os.getcwd())
from evita.execution import ExecutionResult, AgentOutcome, Plan
from evita.executors.frenetix_executor import read_frenetix_plans
from evita.evaluators.frenetix_evaluator import FrenetixScenarioEvaluator
from evita.data_analysis import _rebuild_trajectory

results = []
def check(name, ok):
    results.append(ok); print(("PASS " if ok else "FAIL ") + name)

def old_planned_trajectories(simulation):
    """The evaluator's _planned_trajectories before step 2 (code_4.zip), copied verbatim"""
    rows = []
    for agent_id in simulation.agent_id_list:
        agent_log = pd.read_csv(os.path.join(simulation.config.simulation.log_path, str(agent_id), "logs.csv"), sep = ';')
        for record in sorted(agent_log.to_dict(orient = "records"), key = lambda r: int(r["trajectory_number"])):
            timestep = int(record["trajectory_number"])
            rows.append([agent_id, timestep, _rebuild_trajectory(timestep, record["x_positions_m"], record["y_positions_m"], record["theta_orientations_rad"], record["velocities_mps"])])
    return pd.DataFrame(rows, columns = ["vehicle_id", "timestep", "trajectory"])

# Agents' logs.csv in FrenetiX's format (rows deliberately out of order)
log_path = tempfile.mkdtemp(); rng = np.random.default_rng(1)
agent_ids = [10003, 10001, 10002]
for agent_id in agent_ids:
    os.makedirs(os.path.join(log_path, str(agent_id)))
    rows = []
    for step in rng.permutation(40):
        n = 30
        rows.append({"trajectory_number": int(step), "x_positions_m": ",".join(f"{v:.6f}" for v in rng.normal(size = n)),
                     "y_positions_m": ",".join(f"{v:.6f}" for v in rng.normal(size = n)), "theta_orientations_rad": ",".join(f"{v:.6f}" for v in rng.normal(size = n)),
                     "velocities_mps": ",".join(f"{v:.6f}" for v in rng.uniform(0, 10, n)), "other_column": 1})
    pd.DataFrame(rows).to_csv(os.path.join(log_path, str(agent_id), "logs.csv"), sep = ";", index = False)
simulation = SimpleNamespace(agent_id_list = agent_ids, config = SimpleNamespace(simulation = SimpleNamespace(log_path = log_path)))

def same_tables(a, b):
    if list(a["vehicle_id"]) != list(b["vehicle_id"]) or list(a["timestep"]) != list(b["timestep"]):
        return False
    for ta, tb in zip(a["trajectory"], b["trajectory"]):
        if ta.initial_time_step != tb.initial_time_step or len(ta.state_list) != len(tb.state_list):
            return False
        for sa, sb in zip(ta.state_list, tb.state_list):
            if sa.time_step != sb.time_step or not np.array_equal(sa.position, sb.position) or sa.orientation != sb.orientation or sa.velocity != sb.velocity:
                return False
    return True

old = old_planned_trajectories(simulation)
agents = {agent_id: AgentOutcome(id = agent_id, initial_time_step = 0, status = "COMPLETED_SUCCESS", last_timestep = 40) for agent_id in agent_ids}
execution = ExecutionResult(scenario_id = 1, output_folder = log_path, executed_scenario = None, agents = agents, simulator = "frenetix",
                            plan_loader = functools.partial(read_frenetix_plans, log_path, list(agent_ids)))
restored = pickle.loads(pickle.dumps(execution))
check("ExecutionResult with its plan loader survives pickling", restored.plan_loader is not None)
new = FrenetixScenarioEvaluator._plans_table(restored.load_plans())
check(f"new path gives the same table as the old _planned_trajectories ({len(old)} plans, same order, same states)", same_tables(old, new))
check("plans set on each AgentOutcome, in time order", all([p.time_step for p in restored.agents[a].plans] == list(range(40)) for a in agent_ids))
check("plans loaded once (second call returns the same objects)", restored.load_plans() is restored.load_plans())
check("FrenetiX plans carry no decisions", all(p.object_decisions is None for a in agent_ids for p in restored.agents[a].plans))
check("_planned_trajectories still works with a Simulation stand-in (false_positive_analysis.py)",
      same_tables(old, FrenetixScenarioEvaluator._planned_trajectories(simulation)))
preset = ExecutionResult(scenario_id = 2, output_folder = "", executed_scenario = None,
                         agents = {1: AgentOutcome(id = 1, initial_time_step = 0, status = "COMPLETED_SUCCESS", last_timestep = 1, plans = [Plan(0, _rebuild_trajectory(0, "1", "2", "0", "1"))])})
check("without a loader, load_plans returns the plans already set (simulators that fill them directly)", list(preset.load_plans()) == [1])

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
