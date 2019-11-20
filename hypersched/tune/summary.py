import sys
import os
import numpy as np


class Summary:
    def __init__(self, metric, mode="max"):
        self._metric = metric
        self._trials = set()
        self._best = {}
        if mode == "max":
            self._op = lambda a, b: a.get(metric, -np.inf) > b.get(
                metric, -np.inf
            )
        else:
            raise ValueError

    def on_result(self, trial, result):
        self._trials.add(trial)
        if self._op(result, self._best):
            self._best = result.copy()

    def summarize(self, params, fname=None, mode="csv"):
        lengths = [
            t.last_result.get("training_iteration", 0) for t in self._trials
        ]

        results = params.copy()
        results.update(
            {
                "directory": list(self._trials)[0].local_dir,
                "num_trials": len(self._trials),
                "metric": self._metric,
                "best": self._best.get(self._metric),
                "best_iter": self._best.get("training_iteration"),
                "avg_len": np.mean(lengths),
                "median_len": np.median(lengths),
                "top_10_len": np.percentile(lengths, 90),
                "bottom_10_len": np.percentile(lengths, 10),
            }
        )
        sum_str = []
        sum_str += ["=" * 20]
        sum_str += [f"Result for {' '.join(sys.argv)}"]
        sum_str += ["Results written at " + results["directory"]]
        sum_str += [f"Trials seen: {results['num_trials']}"]
        sum_str += [f"Best {self._metric}: {results['best']:0.4f}"]
        sum_str += [f"Seen at iter: {results['best_iter']}"]
        sum_str += [f"Avg Length: {results['avg_len']}"]

        top = sorted(
            list(self._trials),
            key=lambda t: t.last_result.get(self._metric, -np.inf),
        )
        top_3_iters = [
            t.last_result.get("training_iteration") for t in top[-3:]
        ]
        sum_str += [f"Lengths of top scores: {top_3_iters}"]
        top_3_results = [t.last_result.get(self._metric) for t in top[-3:]]
        sum_str += [f"Scores of top scores: {top_3_results}"]
        sum_str += ["=" * 20]
        final = "\n".join(sum_str)
        print(final)
        if fname:
            with open(fname, "a") as f:
                f.write(final)

        if mode == "csv":
            import csv

            write_header = False
            if not os.path.exists(fname + ".csv"):
                write_header = True
            with open(fname + ".csv", "a") as f:
                w = csv.DictWriter(f, results.keys())
                if write_header:
                    w.writeheader()
                w.writerow(results)
