import matplotlib.pyplot as plt
import argparse
import glob
import numpy as np
import pandas as pd
import sys
import os
from utils import timestring
import ast

parser = argparse.ArgumentParser()
parser.add_argument(
    "--logdir",
    "-l",
    type=str,
    default=os.path.expanduser("~/nips_results/baselines/"),
)
parser.add_argument("--metric", "-m", type=str, default="mean_accuracy")
parser.add_argument(
    "--wall", "-w", action="store_true", help="Use wallclock time."
)
parser.add_argument("--show", "-s", action="store_true")
parser.add_argument("--save", action="store_true")


def plot_scheduler(logdir, metric, save=False, use_wall=False, ax=plt):
    max_dir = max(glob.glob(logdir), key=os.path.getmtime)
    paths = glob.glob(os.path.join(max_dir, "*/progress.csv"), recursive=True)
    print("Detected {} paths".format(len(paths)))
    trial_time = "timestamp" if use_wall else "training_iteration"
    experiment_time = "timestamp" if use_wall else "training_iteration"
    max_metrics = []
    end_time = 0
    trial_times = []
    for i, path in enumerate(paths):
        try:
            df = pd.DataFrame.from_csv(path).reset_index()
            ax.plot(df[experiment_time], df[metric])
            # use ast instead of json bc quote handling may be messed up
            max_metrics += [
                (df[metric].max(), ast.literal_eval(df.config[0]).get("seed"))
            ]
            trial_times += [df[trial_time].max()]
            end_time = max(end_time, df[experiment_time].max())
        except pd.errors.EmptyDataError:
            pass

    metric_string = "\n".join(
        [f"{m:.4f}" for m, trial_id in sorted(max_metrics, reverse=True)[:5]]
    )
    print(f"Duration: {end_time} - Avg. Dur: {np.mean(trial_times)}")
    print(f"Here are the top 5 scores:\n{metric_string}")
    import csv

    if save:
        with open("results.csv", mode="a") as file:
            writer = csv.writer(
                file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            lst = [str(end_time), str(np.mean(trial_times))]
            lst += [
                str(m) for m, trial_id in sorted(max_metrics, reverse=True)[:3]
            ]

            writer.writerow(lst)


# plots the most recently created results
if __name__ == "__main__":
    args = parser.parse_args()
    plot_scheduler(args.logdir, args.metric, save=args.save, use_wall=args.wall)
    plt.xlabel("wall" if args.wall else "iterations")
    plt.ylabel(args.metric)

    # plt.savefig("figures/figure" + timestring() + ".png")
    if not os.path.exists("figures/"):
        os.makedirs("figures/")

    if args.show:
        plt.savefig(
            "figures/"
            + os.path.basename(args.logdir)
            + "{}.png".format(timestring())
        )
        plt.show()
