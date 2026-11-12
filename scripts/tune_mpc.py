"""Measure dry-track completion at explicit speed caps; never hide failed runs."""
import json
from pathlib import Path

from racer.controllers.mpc import MPCController
from racer.raceline import Raceline
from racer.runner import run_episode


def main():
    rl = Raceline.load()
    rows = []
    for cap, weight in [(cap, 1000) for cap in [0.97, 0.9, 0.8, 0.7]]:
        for adapted in [False, True]:
            controller = MPCController(rl, speed_cap=cap, w_slack=weight, understeer_compensation=True)
            if adapted:
                controller.set_grip(controller.p.mu)
            result = run_episode(controller, max_sim_time=240)
            row = dict(speed_cap=cap, slack_weight=weight, diagnostic_only=(weight == 0), adapted=adapted, completed=result.completed,
                       crashed=result.crashed, lap_time=result.lap_time,
                       max_cross_track=result.max_cross_track,
                       solve_fails=controller.solve_fails)
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path('results/mpc_calibration.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
