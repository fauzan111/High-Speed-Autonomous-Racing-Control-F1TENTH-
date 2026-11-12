"""Plot saved experiment logs: python -m scripts.plot_grip results/grip_wet."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def plot_experiment(directory: Path) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    try:
        for variant in ['fixed_dry', 'fixed_wet', 'adaptive']:
            with (directory / f'{variant}.csv').open() as source:
                rows = list(csv.DictReader(source))
            if not rows:
                continue
            ts = [float(r['time']) for r in rows]
            axes[1].plot(ts, [float(r['speed']) for r in rows], label=variant)
            axes[2].plot(ts, [float(r['cross_track']) for r in rows], label=variant)
            if variant == 'adaptive':
                mu = [float(r['mu_hat']) for r in rows]
                sigma = [float(r['sigma']) for r in rows]
                axes[0].plot(ts, [float(r['true_mu']) for r in rows], color='black', label='true mu')
                axes[0].plot(ts, mu, label='estimated mu')
                axes[0].fill_between(ts, [m-2*s for m,s in zip(mu,sigma)],
                                     [m+2*s for m,s in zip(mu,sigma)], alpha=0.2, label='±2 sigma (heuristic)')
                axes[0].plot(ts, [float(r['mu_used']) for r in rows], label='MPC mu used')
        for ax, label in zip(axes, ['Friction coefficient', 'Speed [m/s]', 'Tracking error [m]']):
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend(loc='best')
        axes[-1].set_xlabel('Simulation time [s] (standing start and flying lap)')
        fig.suptitle('F1TENTH linear tyre-model parameter estimation')
        fig.tight_layout()
        output = directory / 'comparison.png'
        fig.savefig(output, dpi=160)
        return output
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(plot_experiment(parser.parse_args().directory))


if __name__ == '__main__':
    main()
