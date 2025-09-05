
from lib_piglet.utils.tools import eprint
from typing import List, Tuple, Dict, Optional
import glob, os, sys, time, json, heapq, copy

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)




#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0

# Logging configuration
LOG_FILE = os.path.join(os.path.dirname(__file__), "planning_diagnostics.log")
RUN_ID = int(time.time())

def _log_write(line: str):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"{ts} [RUN {RUN_ID}] {line}\n")
    except Exception:
        pass

def log_event(tag: str, payload: dict):
    try:
        _log_write(f"{tag} | {json.dumps(payload, ensure_ascii=False)}")
    except Exception:
        _log_write(f"{tag} | {payload}")

#########################
# Reimplementing the content in get_path() function and replan() function.
#
# They both return a list of paths. A path is a list of (x,y) location tuples.
# The path should be conflict free.
# Hint, you could use some global variables to reuse many resources across get_path/replan function calls.
#########################


# This function return a list of location tuple as the solution.
# @param env The flatland railway environment
# @param agents A list of EnvAgent.
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    ############
    # Multi-Agent Prioritized Space-Time A* with Reservation Table
    # - Plans all agents sequentially by priority (deadline slack first)
    # - Uses time-expanded A* to avoid vertex/edge conflicts
    # - Pads paths to episode length to avoid early termination
    ############

    # Helpers
    def in_bounds(pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < rail.height and 0 <= y < rail.width

    def next_pos_from_dir(pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        x, y = pos
        if direction == Directions.NORTH:
            return (x - 1, y)
        if direction == Directions.EAST:
            return (x, y + 1)
        if direction == Directions.SOUTH:
            return (x + 1, y)
        if direction == Directions.WEST:
            return (x, y - 1)
        return pos

    def neighbors(pos: Tuple[int, int], direction: int) -> List[Tuple[Tuple[int, int], int]]:
        trans = rail.get_transitions(pos[0], pos[1], direction)
        nbrs = []
        for dir_out in range(4):
            if trans[dir_out]:
                npos = next_pos_from_dir(pos, dir_out)
                if in_bounds(npos):
                    nbrs.append((npos, dir_out))
        return nbrs

    # Precompute minimal rail distance (ignoring other agents) for prioritization
    def min_steps_ignore_conflict(start: Tuple[int, int], start_dir: int, goal: Tuple[int, int]) -> int:
        # BFS on (pos, dir)
        from collections import deque
        q = deque()
        q.append((start, start_dir, 0))
        visited = set([(start, start_dir)])
        while q:
            pos, d, dist = q.popleft()
            if pos == goal:
                return dist
            for (npos, nd) in neighbors(pos, d):
                state = (npos, nd)
                if state not in visited:
                    visited.add(state)
                    q.append((npos, nd, dist + 1))
        # If unreachable, return a large number to de-prioritize
        return max_timestep

    # Reservation tables: vertex and edge reservations over time
    # vertex_res[t] -> set of (x,y)
    # edge_res[t] -> set of ((x1,y1),(x2,y2)) traversed from t-1 -> t
    vertex_res: Dict[int, set] = {}
    edge_res: Dict[int, set] = {}

    def reserve_path(path: List[Tuple[int, int]], start_t: int = 0):
        # Reserve vertices at indices >= start_t and edges between consecutive indices
        for t in range(max(start_t + 1, 1), len(path)):
            # Reserve vertex at time t
            vertex_res.setdefault(t, set()).add(path[t])
            # Reserve edge for move path[t-1] -> path[t]
            edge_res.setdefault(t, set()).add((path[t - 1], path[t]))
        # Also reserve t=0 vertex to reduce initial spawn conflicts
        if len(path) > 0:
            vertex_res.setdefault(0, set()).add(path[0])

    def is_reserved_vertex(pos: Tuple[int, int], t: int) -> bool:
        return pos in vertex_res.get(t, set())

    def is_reserved_edge(u: Tuple[int, int], v: Tuple[int, int], t: int) -> bool:
        # Edge reservation stored for traversal arriving at time t (from t-1)
        return (u, v) in edge_res.get(t, set())

    def plan_single_agent(agent_id: int, start_pos: Tuple[int, int], start_dir: int, goal: Tuple[int, int],
                          start_time_idx: int) -> Tuple[List[Tuple[int, int]], Dict]:
        # A* in time-expanded graph over state (t, pos, dir)
        # path indices follow: path[0] is initial position at t=0
        # We will construct suffix starting at t=start_time_idx and then merge

        # Heuristic: Manhattan distance (admissible lower bound on rail length; not perfect but ok)
        def h(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue: (f, g=t, t, pos, dir)
        pq: List[Tuple[int, int, int, Tuple[int, int], int]] = []
        start_state = (start_time_idx, start_pos, start_dir)
        heapq.heappush(pq, (start_time_idx + h(start_pos), start_time_idx, start_time_idx, start_pos, start_dir))
        # best[(pos, dir, t)] = g
        best: Dict[Tuple[Tuple[int, int], int, int], int] = {}
        parent: Dict[Tuple[int, Tuple[int, int], int], Tuple[int, Tuple[int, int], int]] = {}

        max_search_t = max_timestep  # bound by episode
        expanded = 0
        pushed = 1
        pruned_vertex = 0
        pruned_edge = 0

        while pq:
            f, g, t, pos, d = heapq.heappop(pq)
            expanded += 1
            state_key = (pos, d, t)
            if best.get(state_key, sys.maxsize) < g:
                continue

            # Goal reached: construct suffix path from start_time_idx to t
            if pos == goal:
                # Reconstruct positions per timestep from start_time_idx..t
                rev = []
                cur = (t, pos, d)
                while True:
                    rev.append((cur[1][0], cur[1][1]))
                    if cur not in parent:
                        break
                    cur = parent[cur]
                rev.reverse()
                stats = {
                    "agent": agent_id,
                    "expanded": expanded,
                    "pushed": pushed,
                    "pruned_vertex": pruned_vertex,
                    "pruned_edge": pruned_edge,
                    "start_t": start_time_idx,
                    "goal_reached_t": t,
                    "path_len": len(rev)
                }
                return rev, stats  # includes position at start_time_idx as first element

            if t >= max_search_t:
                continue

            # 1) Wait action (stay)
            nt = t + 1
            if not is_reserved_vertex(pos, nt):
                nkey = (pos, d, nt)
                ng = g + 1
                if ng < best.get(nkey, sys.maxsize):
                    best[nkey] = ng
                    parent[(nt, pos, d)] = (t, pos, d)
                    nf = ng + h(pos)
                    heapq.heappush(pq, (nf, ng, nt, pos, d))
                    pushed += 1
            else:
                pruned_vertex += 1

            # 2) Move actions
            for (npos, nd) in neighbors(pos, d):
                nt = t + 1
                # Check vertex and edge reservations
                if is_reserved_vertex(npos, nt):
                    pruned_vertex += 1
                    continue
                # Edge swap conflict: someone traverses npos->pos at same time
                if is_reserved_edge(npos, pos, nt):
                    pruned_edge += 1
                    continue
                nkey = (npos, nd, nt)
                ng = g + 1
                if ng < best.get(nkey, sys.maxsize):
                    best[nkey] = ng
                    parent[(nt, npos, nd)] = (t, pos, d)
                    nf = ng + h(npos)
                    heapq.heappush(pq, (nf, ng, nt, npos, nd))
                    pushed += 1

        # Failed to find path within horizon; return just staying in place to avoid crashes
        # The caller should handle padding
        stats = {
            "agent": agent_id,
            "expanded": expanded,
            "pushed": pushed,
            "pruned_vertex": pruned_vertex,
            "pruned_edge": pruned_edge,
            "start_t": start_time_idx,
            "goal_reached_t": None,
            "path_len": 1,
            "failed": True
        }
        return [start_pos], stats

    # Priority: sort agents by slack (deadline - min rail steps). Smaller slack first
    agents_info = []
    for a in agents:
        start = a.initial_position
        sdir = a.initial_direction
        goal = a.target
        min_len = min_steps_ignore_conflict(start, sdir, goal)
        deadline = getattr(a, 'deadline', None)
        slack = (deadline - min_len) if deadline is not None else 10**9
        agents_info.append((a.handle, start, sdir, goal, min_len, deadline, slack))

    # Sort by slack asc, then by min_len desc (longer first to reduce blocking), then by handle
    agents_info.sort(key=lambda x: (x[6], -x[4], x[0]))

    # Plan sequentially with reservations
    path_all: List[List[Tuple[int, int]]] = [[] for _ in range(len(agents))]
    planning_stats: Dict[int, Dict] = {}

    # Log priorities
    try:
        log_event("PLAN_START", {
            "agents": len(agents),
            "max_timestep": max_timestep,
            "priorities": [
                {
                    "agent": aid,
                    "start": start,
                    "start_dir": sdir,
                    "goal": goal,
                    "deadline": _ddl,
                    "min_len": _min_len,
                    "slack": _slack
                } for (aid, start, sdir, goal, _min_len, _ddl, _slack) in agents_info
            ]
        })
    except Exception:
        pass

    for (aid, start, sdir, goal, _min_len, _ddl, _slack) in agents_info:
        # Plan suffix from t=0; Build initial stub with start at index 0
        suffix, stats = plan_single_agent(aid, start, sdir, goal, 0)
        planning_stats[aid] = stats
        if len(suffix) == 0:
            # Fallback: at least stay in start
            suffix = [start]

        # Ensure the path begins with the start position at t=0
        if suffix[0] != start:
            suffix = [start] + suffix

        path = suffix[:]
        # If reached goal, keep staying at goal to avoid premature finish
        if path[-1] != goal:
            # Try naive extension by greedy moves until goal if possible
            # (rare case when A* returned only partial wait path)
            pass

        # Pad to episode length
        last = path[-1]
        while len(path) < max_timestep:
            path.append(last)

        path_all[aid] = path
        # Reserve this path to avoid conflicts for subsequent agents
        reserve_path(path)

    # Post-planning diagnostics
    try:
        # Compute simple congestion metrics and wait ratios up to first goal arrival
        top_cells: Dict[Tuple[int, int], int] = {}
        wait_ratios: Dict[int, float] = {}
        for a in agents:
            aid = a.handle
            p = path_all[aid]
            # find first arrival index
            goal_idx = None
            try:
                goal_idx = p.index(a.target)
            except ValueError:
                goal_idx = min(len(p), max_timestep) - 1

            moves = 0
            waits = 0
            prev = None
            for t in range(0, min(goal_idx + 1, len(p))):
                pos = p[t]
                top_cells[pos] = top_cells.get(pos, 0) + 1
                if prev is not None:
                    if pos == prev:
                        waits += 1
                    else:
                        moves += 1
                prev = pos
            total = max(1, moves + waits)
            wait_ratios[aid] = round(waits / total, 3)

        busiest = sorted(top_cells.items(), key=lambda kv: kv[1], reverse=True)[:10]
        suggestions = []
        high_wait_agents = [aid for aid, wr in wait_ratios.items() if wr >= 0.5]
        if len(high_wait_agents) >= max(1, len(agents) // 5):
            suggestions.append("High waiting ratios detected; consider tuning priority ordering or enabling longer detours.")
        tight_slack = [info[0] for info in agents_info if info[6] is not None and info[6] <= 2]
        if len(tight_slack) > 0:
            suggestions.append("Tight deadline slack for some agents; prioritize them more aggressively or reduce blocking early.")
        heavy_pruning = [aid for aid, s in planning_stats.items() if s.get("pruned_vertex", 0) + s.get("pruned_edge", 0) > 1000]
        if len(heavy_pruning) > 0:
            suggestions.append("Many states pruned due to reservations; map is congested. Try alternative routes or batch replanning.")

        log_event("PLAN_SUMMARY", {
            "per_agent_stats": planning_stats,
            "wait_ratio": wait_ratios,
            "busiest_cells": busiest,
            "suggestions": suggestions
        })
    except Exception:
        pass

    return path_all

# This function return a list of location tuple as the solution.
# @param rail The flatland railway GridTransitionMap
# @param agents A list of EnvAgent.
# @param current_timestep The timestep that malfunction/collision happens .
# @param existing_paths The existing paths from previous get_plan or replan.
# @param max_timestep The max timestep of this episode.
# @param new_malfunction_agents  The id of agents have new malfunction happened at current time step (Does not include agents already have malfunciton in past timesteps)
# @param failed_agents  The id of agents failed to reach the location on its path at current timestep.
# @return path_all  Return paths that locaitons from current_timestp is updated to handle malfunctions and failed execuations.
def replan(agents: List[EnvAgent], rail: GridTransitionMap, current_timestep: int, existing_paths: List[Tuple],
           max_timestep: int, new_malfunction_agents: List[int], failed_agents: List[int]):
    ############
    # Dynamic Replanning Using Reservations from Existing Paths
    # - Replan only affected agents (new malfunctions or failed execution)
    # - Keep unaffected agents' future as reservations to minimize ripple
    # - Enforce waiting for malfunction duration
    ############

    # Helpers shared with get_path
    def in_bounds(pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < rail.height and 0 <= y < rail.width

    def next_pos_from_dir(pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        x, y = pos
        if direction == Directions.NORTH:
            return (x - 1, y)
        if direction == Directions.EAST:
            return (x, y + 1)
        if direction == Directions.SOUTH:
            return (x + 1, y)
        if direction == Directions.WEST:
            return (x, y - 1)
        return pos

    def neighbors(pos: Tuple[int, int], direction: int) -> List[Tuple[Tuple[int, int], int]]:
        trans = rail.get_transitions(pos[0], pos[1], direction)
        nbrs = []
        for dir_out in range(4):
            if trans[dir_out]:
                npos = next_pos_from_dir(pos, dir_out)
                if in_bounds(npos):
                    nbrs.append((npos, dir_out))
        return nbrs

    def min_steps_ignore_conflict(start: Tuple[int, int], start_dir: int, goal: Tuple[int, int]) -> int:
        from collections import deque
        q = deque()
        q.append((start, start_dir, 0))
        visited = set([(start, start_dir)])
        while q:
            pos, d, dist = q.popleft()
            if pos == goal:
                return dist
            for (npos, nd) in neighbors(pos, d):
                state = (npos, nd)
                if state not in visited:
                    visited.add(state)
                    q.append((npos, nd, dist + 1))
        return max_timestep

    # Build reservation from unaffected agents for t >= current_timestep+1
    vertex_res: Dict[int, set] = {}
    edge_res: Dict[int, set] = {}

    def reserve_future(path: List[Tuple[int, int]]):
        # Reserve vertices and edges from current_timestep+1 onwards
        for t in range(max(current_timestep + 1, 1), len(path)):
            vertex_res.setdefault(t, set()).add(path[t])
            edge_res.setdefault(t, set()).add((path[t - 1], path[t]))

    def is_reserved_vertex(pos: Tuple[int, int], t: int) -> bool:
        return pos in vertex_res.get(t, set())

    def is_reserved_edge(u: Tuple[int, int], v: Tuple[int, int], t: int) -> bool:
        return (u, v) in edge_res.get(t, set())

    # Copy existing paths to modify
    new_paths: List[List[Tuple[int, int]]] = [list(p) for p in existing_paths]

    # Identify replanning set
    replan_set = set(new_malfunction_agents) | set(failed_agents)

    # Log replan start
    try:
        log_event("REPLAN_START", {
            "current_timestep": current_timestep,
            "new_malfunctions": list(new_malfunction_agents),
            "failed_agents": list(failed_agents),
            "replan_set": list(replan_set)
        })
    except Exception:
        pass

    # Reserve unaffected agents
    for aid in range(len(agents)):
        if aid in replan_set:
            continue
        if aid < len(new_paths):
            reserve_future(new_paths[aid])

    # Prioritize replanning agents by slack
    prio = []  # (slack, -min_len, aid)
    for aid in replan_set:
        a = agents[aid]
        cur_pos = a.position if a.position is not None else a.initial_position
        cur_dir = a.direction if a.position is not None else a.initial_direction
        goal = a.target
        min_len = min_steps_ignore_conflict(cur_pos, cur_dir, goal)
        deadline = getattr(a, 'deadline', None)
        slack = (deadline - min_len) if deadline is not None else 10**9
        prio.append((slack, -min_len, aid))
    prio.sort()

    # Plan each replanning agent sequentially considering reservations
    replanning_stats: Dict[int, Dict] = {}
    for _slack, _neg_len, aid in prio:
        a = agents[aid]
        cur_pos = a.position if a.position is not None else a.initial_position
        cur_dir = a.direction if a.position is not None else a.initial_direction
        goal = a.target

        # If already done, just pad and continue
        if getattr(a, 'status', None) in [2, 3]:
            # Ensure padding for safety
            last = new_paths[aid][-1] if len(new_paths[aid]) > 0 else cur_pos
            while len(new_paths[aid]) < max_timestep:
                new_paths[aid].append(last)
            continue

        # Forced wait time due to new malfunction
        forced_wait = a.malfunction_data["malfunction"] if aid in new_malfunction_agents else 0

        # Build prefix up to current_timestep intact
        base_prefix = new_paths[aid][:current_timestep + 1]
        if len(base_prefix) == 0:
            base_prefix = [cur_pos]
        # Ensure the prefix ends at current position to avoid inconsistencies
        if base_prefix[-1] != cur_pos:
            base_prefix[-1] = cur_pos

        # Reserve own forced waiting on current cell to prevent others colliding in
        for t in range(current_timestep + 1, min(current_timestep + 1 + forced_wait, max_timestep)):
            vertex_res.setdefault(t, set()).add(cur_pos)
            edge_res.setdefault(t, set()).add((cur_pos, cur_pos))

        # Plan suffix from depart_time = current_timestep + forced_wait
        depart_t = min(current_timestep + forced_wait, max_timestep - 1)

        # A* planning from (cur_pos, cur_dir) at time depart_t
        def h(p: Tuple[int, int]) -> int:
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        pq: List[Tuple[int, int, int, Tuple[int, int], int]] = []
        heapq.heappush(pq, (depart_t + h(cur_pos), depart_t, depart_t, cur_pos, cur_dir))
        best: Dict[Tuple[Tuple[int, int], int, int], int] = {}
        parent: Dict[Tuple[int, Tuple[int, int], int], Tuple[int, Tuple[int, int], int]] = {}

        found_suffix: Optional[List[Tuple[int, int]]] = None
        expanded = 0
        pushed = 1
        pruned_vertex = 0
        pruned_edge = 0

        while pq:
            f, g, t, pos, d = heapq.heappop(pq)
            expanded += 1
            key = (pos, d, t)
            if best.get(key, sys.maxsize) < g:
                continue
            if pos == goal:
                # Reconstruct suffix from depart_t..t
                rev = []
                cur = (t, pos, d)
                while True:
                    rev.append((cur[1][0], cur[1][1]))
                    if cur not in parent:
                        break
                    cur = parent[cur]
                rev.reverse()
                found_suffix = rev
                break
            if t >= max_timestep - 1:
                continue

            # 1) Wait
            nt = t + 1
            if not is_reserved_vertex(pos, nt):
                nkey = (pos, d, nt)
                ng = g + 1
                if ng < best.get(nkey, sys.maxsize):
                    best[nkey] = ng
                    parent[(nt, pos, d)] = (t, pos, d)
                    nf = ng + h(pos)
                    heapq.heappush(pq, (nf, ng, nt, pos, d))
                    pushed += 1
            else:
                pruned_vertex += 1

            # 2) Move
            for (npos, nd) in neighbors(pos, d):
                nt = t + 1
                if is_reserved_vertex(npos, nt):
                    pruned_vertex += 1
                    continue
                if is_reserved_edge(npos, pos, nt):
                    pruned_edge += 1
                    continue
                nkey = (npos, nd, nt)
                ng = g + 1
                if ng < best.get(nkey, sys.maxsize):
                    best[nkey] = ng
                    parent[(nt, npos, nd)] = (t, pos, d)
                    nf = ng + h(npos)
                    heapq.heappush(pq, (nf, ng, nt, npos, nd))
                    pushed += 1

        # Construct new path for this agent
        new_path = base_prefix[:]
        # Inject forced waits (if any) between current_timestep+1 .. depart_t
        for t in range(current_timestep + 1, depart_t + 1):
            new_path.append(cur_pos)

        if found_suffix is None:
            # If search failed, stay put afterwards
            if len(new_path) == 0:
                new_path = [cur_pos]
            last = new_path[-1]
            while len(new_path) < max_timestep:
                new_path.append(last)
        else:
            # found_suffix includes position at depart_t as first element
            # Merge suffix, avoid duplicating the depart position if already appended
            if len(new_path) > 0 and new_path[-1] == found_suffix[0]:
                new_path.extend(found_suffix[1:])
            else:
                new_path.extend(found_suffix)
            # Pad to episode length
            last = new_path[-1]
            while len(new_path) < max_timestep:
                new_path.append(last)

        new_paths[aid] = new_path

        # Reserve this replanned path for subsequent replanning agents
        for t in range(max(current_timestep + 1, 1), len(new_path)):
            vertex_res.setdefault(t, set()).add(new_path[t])
            edge_res.setdefault(t, set()).add((new_path[t - 1], new_path[t]))

        # Log per-agent replanning result
        try:
            replanning_stats[aid] = {
                "agent": aid,
                "cur_pos": cur_pos,
                "cur_dir": cur_dir,
                "goal": goal,
                "forced_wait": forced_wait,
                "expanded": expanded,
                "pushed": pushed,
                "pruned_vertex": pruned_vertex,
                "pruned_edge": pruned_edge,
                "path_len": len(new_path),
                "depart_t": depart_t,
                "goal_reached": goal in new_path
            }
        except Exception:
            pass

    # Replan summary
    try:
        # Summarize and suggest improvements
        wait_ratios: Dict[int, float] = {}
        for aid in replan_set:
            p = new_paths[aid]
            goal = agents[aid].target
            goal_idx = None
            try:
                goal_idx = p.index(goal)
            except ValueError:
                goal_idx = min(len(p), max_timestep) - 1
            moves = 0
            waits = 0
            for t in range(current_timestep + 1, goal_idx + 1):
                if t <= 0 or t >= len(p):
                    break
                if p[t] == p[t - 1]:
                    waits += 1
                else:
                    moves += 1
            total = max(1, moves + waits)
            wait_ratios[aid] = round(waits / total, 3)

        suggestions = []
        high_wait = [aid for aid, wr in wait_ratios.items() if wr >= 0.5]
        if len(high_wait) > 0:
            suggestions.append("Replanned agents waiting a lot; try alternative corridors or earlier diversion.")
        heavy_pruning = [aid for aid, s in replanning_stats.items() if s.get("pruned_vertex", 0) + s.get("pruned_edge", 0) > 500]
        if len(heavy_pruning) > 0:
            suggestions.append("Replanning heavily constrained by reservations; consider temporary relaxation or batch replan.")

        log_event("REPLAN_SUMMARY", {
            "current_timestep": current_timestep,
            "affected": list(replan_set),
            "per_agent_stats": replanning_stats,
            "wait_ratio": wait_ratios,
            "suggestions": suggestions
        })
    except Exception:
        pass

    return new_paths


#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv, replan = replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))

        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        deadline_files =  [test.replace(".pkl",".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan = replan)
