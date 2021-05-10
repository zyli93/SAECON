import glob
from tqdm import tqdm

BASELINE = {
    "BETTER": 0.7821,
    "WORSE": 0.5872,
    "NONE": 0.9298,
    "micro": 0.8743
}

class Run:
    def __init__(
        self, 
        name,
        val_F1_better,
        val_F1_worse,
        val_F1_none,
        val_F1_micro
        ):
        self.name=name
        self.val_F1_better = val_F1_better
        self.val_F1_worse = val_F1_worse
        self.val_F1_none = val_F1_none
        self.val_F1_micro = val_F1_micro

        self.val_F1_better_best = max(self.val_F1_better) if self.val_F1_better else -1
        self.val_F1_worse_best = max(self.val_F1_worse) if self.val_F1_worse else -1
        self.val_F1_none_best = max(self.val_F1_none) if self.val_F1_none else -1
        self.val_F1_micro_best = max(self.val_F1_micro) if self.val_F1_micro else -1

        self.performant = self.val_F1_better_best > BASELINE["BETTER"] and \
            self.val_F1_worse_best > BASELINE["WORSE"] and \
            self.val_F1_none_best > BASELINE["NONE"] and \
            self.val_F1_micro_best > BASELINE["micro"]

    @classmethod
    def from_file(cls, file_path):
        val_F1_better = []
        val_F1_worse = []
        val_F1_none = []
        val_F1_micro = []

        with open(file_path) as fp:
            lines = fp.readlines()

        for line in lines: 
            if "[Perf-CPC][val][epoch]" in line:
                i_better = line.find("BETTER-") + 7
                i_worse = line.find("WORSE-") + 6
                i_none = line.find("NONE-") + 5
                i_micro = line.find("micro-") + 6

                val_F1_better.append(float(line[i_better:i_better+8]))
                val_F1_worse.append(float(line[i_worse:i_worse+8]))
                val_F1_none.append(float(line[i_none:i_none+8]))
                val_F1_micro.append(float(line[i_micro:i_micro+8]))

        return cls(file_path, val_F1_better, val_F1_worse, val_F1_none, val_F1_micro) 
        
if __name__ == "__main__":
    runs = []
    for file_path in tqdm(glob.glob("log/grid_ratio_*")):
        runs.append(Run.from_file(file_path))
    
    f1_better_best, f1_better_best_i = max((r.val_F1_better_best, i) for i, r in enumerate(runs))
    f1_worse_best, f1_worse_best_i = max((r.val_F1_worse_best, i) for i, r in enumerate(runs))
    f1_none_best, f1_none_best_i = max((r.val_F1_none_best, i) for i, r in enumerate(runs))
    f1_micro_best, f1_micro_best_i = max((r.val_F1_micro_best, i) for i, r in enumerate(runs))

    print("best F1 BETTER: ", f1_better_best, runs[f1_better_best_i].name)
    print("best F1 WORSE: ", f1_worse_best, runs[f1_better_best_i].name)
    print("best F1 NONE: ", f1_none_best, runs[f1_none_best_i].name)
    print("best F1 micro: ", f1_micro_best, runs[f1_micro_best_i].name)