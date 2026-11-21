"""
Check the new Apollo conversion (evita/apollo) on real data, without running EVITA's search.

    python scripts/apollo_check.py trace   <trace.json.gz>
        What Apollo recorded: messages per vehicle, timing, decisions, emergency stops, engage advice, oracles

    python scripts/apollo_check.py convert <trace.json.gz> --map <converted map.xml> [--scenario <scenario.xml>]
                                   [--out executed.xml] [--compare <old executed_scenario.xml>]
        Trace -> executed CommonRoad scenario; facts per vehicle; traffic lights as received vs as specified;
        with --compare, distance between the new and the old executed trajectories.
        EVITA ids: from the trace; otherwise from the generated scenario's planning problems; otherwise (no generated
        scenario) matched in the old executed scenario. Without --scenario, the converted map is the base scenario.

    python scripts/apollo_check.py spec    --scenario <scenario.xml> --map <converted map.xml> [--out c.json]
                                   [--random-lights SEED] [--out-scenario <scenario.xml>] [--compare <old c.json>]
        CommonRoad scenario -> DoppelTest JSON; with --random-lights, the scenario's lights are replaced by DoppelTest
        lights (needed for scenarios generated before the change), and --out-scenario saves that scenario;
        with --compare, differences from the old conversion

Run from the project root, in .venv310.
"""

import os
import re
import sys
import json
import argparse
from collections import Counter

import numpy as np

sys.path.insert(0, os.getcwd())

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet

from evita.apollo.map_data import ApolloMapData
from evita.apollo import traffic_control as tc
from evita.apollo import scenario_converter as spec
from evita.apollo import trace as tr


def object_kind(object_id: str) -> str:
    """
    Kind of a decision's object: Apollo names vehicles "<Apollo id>" or "<Apollo id>_<k>"; other objects "<KIND>_..."
    """
    return "vehicle" if re.fullmatch(r"-?\d+(_\d+)*", object_id) else object_id.split("_")[0]


def stats_ms(values_ns):
    if len(values_ns) < 2:
        return "-"
    gaps = np.diff(np.array(values_ns, dtype = float)) / 1e6
    return f"{np.min(gaps):.0f} / {np.median(gaps):.0f} / {np.max(gaps):.0f} ms"


# ---------- trace ----------

def command_trace(args) -> int:
    trace = tr.read_trace(args.trace)
    scenario = trace.scenario
    print(f"Scenario {scenario.get('generation')}/{scenario.get('scenario')}: status {trace.status} {trace.reason}")
    print(f"Vehicle: {scenario.get('vehicle')}")
    if trace.status != tr.EXECUTED:
        return 0

    t0 = tr.time_zero(trace)
    for vehicle in trace.vehicles:
        times = vehicle.column("t")
        print(f"\n== {vehicle.container}: EVITA id {vehicle.evita_id}, Apollo id {vehicle.apollo_id}, routing {vehicle.routing}")
        if not len(times):
            print("  localization: NONE (this vehicle's Apollo did not run)")
        else:
            print(f"  localization: {len(times)} samples, {(times[-1] - times[0]) / 1e9:.1f} s, intervals min / median / max {stats_ms(times)}; "
                  f"starts {(times[0] - t0) / 1e9:+.2f} s from time step 0")
        plans = vehicle.planning
        print(f"  planning: {len(plans)} messages, intervals {stats_ms([p['t'] for p in plans])}; "
              f"header - record time: median {np.median([(p['t_record'] - p['t']) / 1e6 for p in plans]) if plans else 0:.0f} ms")
        print(f"    main decisions: {dict(Counter(p['main_decision']['decision'] for p in plans))}")
        objects = Counter()
        kinds = Counter()
        for p in plans:
            for obj in p["object_decisions"]:
                kind = object_kind(obj["id"])
                kinds[kind] += 1
                for d in obj["decisions"]:
                    objects[f"{kind}:{d['decision']}"] += 1
        print(f"    object decisions by object kind: {dict(objects)}")
        print(f"    estop in {sum(p['estop']['is_estop'] for p in plans)} plans; engage advice: {dict(Counter(p['engage_advice']['advice'] for p in plans))}; "
              f"is_replan in {sum(p['is_replan'] for p in plans)} plans; trajectory types {dict(Counter(p['trajectory_type'] for p in plans))}")
        print(f"  obstacles: {len(vehicle.obstacles)} messages, intervals {stats_ms([m['t'] for m in vehicle.obstacles])}; "
              f"ids seen {sorted({o['id'] for m in vehicle.obstacles for o in m['obstacles']})}")
        print(f"  traffic lights: {len(vehicle.traffic_lights)} messages, intervals {stats_ms([m['t'] for m in vehicle.traffic_lights])}")
        print(f"  oracles: { {name: len(results) for name, results in vehicle.oracles.items()} }")
    return 0


# ---------- convert ----------

def path_distance(points, path):
    """
    Distance of each point to a polyline, for the points alongside it (points beyond its ends are skipped: the new
    centers are ahead of the old positions, so the new trajectory extends past the old path's end)
    """
    from shapely.geometry import LineString, Point
    if len(path) < 2:
        return np.array([])
    line = LineString(path)
    inside = [Point(p) for p in points if 0.01 < line.project(Point(p)) < line.length - 0.01]
    return np.array([line.distance(point) for point in inside])


def ids_from_old_executed(trace, old_scenario) -> dict:
    """
    EVITA id of each vehicle, from the old executed scenario: the old conversion put each vehicle's first localization
    (Apollo's reference point) as its obstacle's initial position, so each vehicle is matched to the obstacle whose
    initial position is closest to its first localization sample
    """
    candidates = {o.obstacle_id: np.array(o.initial_state.position) for o in old_scenario.dynamic_obstacles}
    matched = {}
    for vehicle in trace.vehicles:
        if not len(vehicle.localization) or not candidates:
            continue
        first = np.array([vehicle.column("x")[0], vehicle.column("y")[0]])
        obstacle_id, position = min(candidates.items(), key = lambda item: np.linalg.norm(item[1] - first))
        if np.linalg.norm(position - first) < 1.0:
            matched[vehicle.container] = obstacle_id
            del candidates[obstacle_id]
    return matched


# Old runs numbered Apollo vehicles from 10000, like the converted maps' traffic lights: such ids are moved by this much
# (10001 -> 20001, EVITA's current convention for Apollo)
OLD_ID_SHIFT = 10000


def used_ids(scenario) -> set:
    """
    Ids already used in a scenario (lanelets, traffic lights and signs, intersections, obstacles)
    """
    network = scenario.lanelet_network
    return (set(getattr(scenario, "_id_set", set())) | {l.lanelet_id for l in network.lanelets}
            | {t.traffic_light_id for t in network.traffic_lights} | {t.traffic_sign_id for t in network.traffic_signs}
            | {i.intersection_id for i in network.intersections})


def shift_colliding_ids(trace, base) -> dict:
    """
    Vehicles whose EVITA id is already used in the base scenario get id + OLD_ID_SHIFT; returns {new id: original id}
    """
    used, original = used_ids(base), {}
    for vehicle in trace.vehicles:
        if vehicle.evita_id is not None and vehicle.evita_id in used:
            original[vehicle.evita_id + OLD_ID_SHIFT] = vehicle.evita_id
            vehicle.evita_id += OLD_ID_SHIFT
    return original


def old_obstacle_of(old_scenario, original_id, vehicle):
    """
    The old executed scenario's obstacle of a vehicle: by its original id, otherwise the obstacle whose initial position
    is closest to the vehicle's first localization (within 1 m), otherwise None
    """
    by_id = {o.obstacle_id: o for o in old_scenario.dynamic_obstacles}
    if original_id in by_id:
        return by_id[original_id]
    if not len(vehicle.localization):
        return None
    first = np.array([vehicle.column("x")[0], vehicle.column("y")[0]])
    candidates = [(np.linalg.norm(np.array(o.initial_state.position) - first), o) for o in old_scenario.dynamic_obstacles]
    distance, closest = min(candidates, key = lambda item: item[0], default = (None, None))
    return closest if closest is not None and distance < 1.0 else None


def prepare_conversion(trace, scenario_xml, map_file, old_executed):
    """
    Base scenario and EVITA ids for a conversion: the generated scenario and its planning problems' ids when available;
    otherwise the converted map (same lanelet network) and the ids matched in the old executed scenario.
    Ids already used by the base scenario (old runs: vehicles from 10000, like the traffic lights) are moved by
    OLD_ID_SHIFT. Returns the base scenario, how the ids were found, and {new id: original id} for the moved ones.
    """
    if scenario_xml:
        base, planning_problems = CommonRoadFileReader(scenario_xml).open()
        ids = list(planning_problems.planning_problem_dict)
        for vehicle in trace.vehicles:
            if vehicle.evita_id is None and vehicle.json_index < len(ids):
                vehicle.evita_id = ids[vehicle.json_index]
        original = shift_colliding_ids(trace, base)
        return base, "planning problems of the generated scenario" + id_note(original), original

    base, _ = CommonRoadFileReader(map_file).open()
    if old_executed:
        old, _ = CommonRoadFileReader(old_executed).open()
        matched = ids_from_old_executed(trace, old)
        for vehicle in trace.vehicles:
            if vehicle.evita_id is None and vehicle.container in matched:
                vehicle.evita_id = matched[vehicle.container]
        original = shift_colliding_ids(trace, base)
        return base, (f"matched in the old executed scenario ({len(matched)} of {sum(len(v.localization) > 0 for v in trace.vehicles)} "
                      f"vehicles that ran)" + id_note(original)), original
    return base, "none (stand-in ids 30000 + k)", {}


def id_note(original: dict) -> str:
    return (f"; ids {sorted(original.values())} were already used in the scenario (old run: vehicles numbered like the "
            f"traffic lights), converted as {sorted(original)}") if original else ""


def command_convert(args) -> int:
    trace = tr.read_trace(args.trace)
    map_data = ApolloMapData.load(args.map)
    generated, ids_source, original_ids = prepare_conversion(trace, args.scenario, args.map, args.compare)
    print(f"Base scenario: {args.scenario or args.map + ' (no generated scenario: the converted map)'}; EVITA ids: {ids_source}")

    conversion = tr.convert_trace(trace, generated, map_data)
    dt = conversion.dt
    print(f"Time step 0: {conversion.t0} ns; {len(conversion.vehicles)} vehicles; {len(conversion.warnings)} warnings")
    for warning in conversion.warnings:
        print(f"  warning: {warning}")

    for vehicle_id, vehicle in conversion.vehicles.items():
        print(f"\n== vehicle {vehicle_id} ({vehicle.container}, Apollo id {vehicle.apollo_id})")
        print(f"  time steps {vehicle.initial_time_step}..{vehicle.last_time_step}; first motion {vehicle.first_motion_step}; "
              f"mission complete {vehicle.mission_complete_step}; {len(vehicle.plans)} plans")
        print(f"  collision entries: {len(vehicle.collision_steps)} (steps {sorted(set(vehicle.collision_steps))[:10]}, "
              f"partners {sorted({p for p in vehicle.collision_partners if p is not None})})")
        print(f"  emergency steps: {sorted(vehicle.emergency_steps)[:10]}; disengagement steps: {sorted(vehicle.disengagement_steps)[:10]}")

    # Traffic lights: as received vs as specified (if the generated scenario's lights were written by traffic_control)
    try:
        specified = tc.TrafficControl.from_commonroad(generated.lanelet_network, dt)
    except tc.TrafficControlError as exc:
        specified = None
        print(f"\nTraffic lights: the generated scenario's lights are not DoppelTest's ({exc}): no comparison")
    if specified is not None:
        last = max((v.last_time_step or 0) for v in conversion.vehicles.values())
        executed_lights = {l.traffic_light_id: l for l in conversion.executed_scenario.lanelet_network.traffic_lights}
        differing = [k for k in range(last + 1)
                     if any(tc.COLOR_OF_STATE.get(executed_lights[i].get_state_at_time_step(k)) != color
                            for i, color in specified.colors_at(k * dt).items())]
        print(f"\nTraffic lights: received colors differ from the specified ones at {len(differing)} of {last + 1} time steps"
              + (f" (first {differing[:8]})" if differing else ""))

    if args.compare:
        old, _ = CommonRoadFileReader(args.compare).open()
        print(f"\nComparison with {args.compare}:")
        for vehicle_id, vehicle in conversion.vehicles.items():
            trace_vehicle = next(v for v in trace.vehicles if v.vehicle_id == vehicle_id)
            old_obstacle = old_obstacle_of(old, original_ids.get(vehicle_id, vehicle_id), trace_vehicle)
            if old_obstacle is None or vehicle.obstacle is None:
                print(f"  vehicle {vehicle_id}: not in both scenarios")
                continue
            old_states = [old_obstacle.initial_state] + (old_obstacle.prediction.trajectory.state_list if old_obstacle.prediction else [])
            new_points = np.array([s.position for s in vehicle.driven_states])
            old_path = np.array([s.position for s in old_states])
            distances = path_distance(new_points, old_path)
            if len(distances) == 0:
                distances = np.array([np.nan])
            start_shift = np.linalg.norm(new_points[0] - old_path[0])
            print(f"  vehicle {vehicle_id}: {len(new_points)} new / {len(old_path)} old states; distance of the new centers to the old "
                  f"path median {np.median(distances):.2f} m, max {np.max(distances):.2f} m; start {start_shift:.2f} m apart "
                  f"(expected about {trace.vehicle.center_offset:.2f} m: the old conversion used Apollo's reference point as the center)")

    if args.out:
        CommonRoadFileWriter(conversion.executed_scenario, PlanningProblemSet()).write_to_file(args.out, OverwriteExistingFile.ALWAYS)
        print(f"\nExecuted scenario written to {args.out}")
    return 0


# ---------- spec ----------

def command_spec(args) -> int:
    scenario, planning_problems = CommonRoadFileReader(args.scenario).open()
    map_data = ApolloMapData.load(args.map)

    if args.random_lights is not None:
        import random
        lights = [l.traffic_light_id for l in scenario.lanelet_network.traffic_lights]
        control = tc.random_traffic_control(map_data, lights, tc.max_duration_green(300, scenario.dt), random.Random(args.random_lights))
        control.to_commonroad(scenario.lanelet_network, scenario.dt)
        print(f"Traffic lights replaced by a random DoppelTest traffic control (seed {args.random_lights})")

    scenario_json = spec.build_scenario_json(scenario, planning_problems, map_data)
    for vehicle_id, adc in zip(scenario_json["evita_vehicle_ids"], scenario_json["ad_section"]["adcs"]):
        print(f"vehicle {vehicle_id}: routing {adc['routing']}, start_s {adc['start_s']:.2f}, dest_s {adc['dest_s']:.2f}, start_t {adc['start_t']}")
    print(f"traffic control: duration_g {scenario_json['tc_section']['duration_g']} s")

    if args.compare:
        old = json.load(open(args.compare))
        print(f"\nComparison with {args.compare} (old conversion):")
        for index, (new, previous) in enumerate(zip(scenario_json["ad_section"]["adcs"], old["ad_section"]["adcs"])):
            print(f"  vehicle {index}: routing {'same' if new['routing'] == previous['routing'] else 'DIFFERENT: ' + str(previous['routing'])}; "
                  f"start_s {new['start_s'] - previous['start_s']:+.2f} m; dest_s {new['dest_s'] - previous['dest_s']:+.2f} m; "
                  f"start_t {new['start_t'] - previous['start_t']:+.2f} s")

    if args.out:
        spec.write_scenario_json(scenario_json, args.out)
        print(f"\nJSON written to {args.out}")
    if args.out_scenario:
        CommonRoadFileWriter(scenario, planning_problems).write_to_file(args.out_scenario, OverwriteExistingFile.ALWAYS)
        print(f"Scenario (with the traffic lights of the JSON) written to {args.out_scenario}")
    return 0


def main(argv) -> int:
    parser = argparse.ArgumentParser(description = "Check the Apollo conversion on real data")
    commands = parser.add_subparsers(dest = "command", required = True)

    p = commands.add_parser("trace"); p.add_argument("trace")
    p = commands.add_parser("convert"); p.add_argument("trace")
    p.add_argument("--scenario", help = "generated scenario (default: the converted map)"); p.add_argument("--map", required = True); p.add_argument("--out")
    p.add_argument("--ids-from-scenario", action = "store_true", help = "(kept for compatibility: ids from the generated scenario are always used)")
    p.add_argument("--compare", help = "old executed_scenario.xml")
    p = commands.add_parser("spec")
    p.add_argument("--scenario", required = True); p.add_argument("--map", required = True); p.add_argument("--out")
    p.add_argument("--random-lights", type = int, help = "seed: replace the scenario's lights by a random DoppelTest traffic control")
    p.add_argument("--out-scenario", help = "save the scenario as converted (with the new lights)")
    p.add_argument("--compare", help = "old c.json")

    args = parser.parse_args(argv)
    return {"trace": command_trace, "convert": command_convert, "spec": command_spec}[args.command](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
