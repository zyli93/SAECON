import datetime
from collections import defaultdict
from itertools import product
import pickle

def dump_pickle(path, obj):
    """ dump object to pickle file """
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)

def parse(path, moi, is_best):
    perf_dict = defaultdict(list)
    """
    moi: metric of interest, can be micro, f1b, f1w, or f1n
    """
    start_time = 0
    with open(path, "r") as fin:
        for i, line in enumerate(fin.readlines()):
            if i == 3:
                starttime = line[1:20]
                starttime_obj = datetime.datetime.strptime(starttime, "%m/%d/%Y %H:%M:%S")

            if "[Perf-CPC][val][epoch]" in line:
                i_better = line.find("BETTER-") + 7
                i_worse = line.find("WORSE-") + 6
                i_none = line.find("NONE-") + 5
                i_micro = line.find("micro-") + 6
                i_time = line[1:20]

                perf_dict['better'].append(float(line[i_better:i_better+8]))
                perf_dict['worse'].append(float(line[i_worse:i_worse+8]))
                perf_dict['none'].append(float(line[i_none:i_none+8]))
                perf_dict['micro'].append(float(line[i_micro:i_micro+8]))
                perf_dict['time'].append(i_time)
    
    if is_best:
        perf = max(perf_dict[moi])
        best_perf_idx = perf_dict[moi].index(perf)
        if best_perf_idx == len(perf_dict[moi]) -1:
            print(f"{path} !")
        best_perf_time_str = perf_dict["time"][best_perf_idx]
        time_obj = datetime.datetime.strptime(best_perf_time_str, '%m/%d/%Y %H:%M:%S')
    else:
        perf = perf_dict[moi][-1]
        last_perf_time_str = perf_dict["time"][-1]
        time_obj = datetime.datetime.strptime(last_perf_time_str, '%m/%d/%Y %H:%M:%S')

    perf_time = time_obj - starttime_obj
    
    return perf, perf_time.seconds

def parse_all(path, is_best):
    perf_dict = defaultdict(list)
    """
    moi: metric of interest, can be micro, f1b, f1w, or f1n
    """
    start_time = 0
    with open(path, "r") as fin:
        for i, line in enumerate(fin.readlines()):
            if i == 3:
                starttime = line[1:20]
                starttime_obj = datetime.datetime.strptime(starttime, "%m/%d/%Y %H:%M:%S")

            if "[Perf-CPC][val][epoch]" in line:
                i_better = line.find("BETTER-") + 7
                i_worse = line.find("WORSE-") + 6
                i_none = line.find("NONE-") + 5
                i_micro = line.find("micro-") + 6
                i_time = line[1:20]

                perf_dict['better'].append(float(line[i_better:i_better+8]))
                perf_dict['worse'].append(float(line[i_worse:i_worse+8]))
                perf_dict['none'].append(float(line[i_none:i_none+8]))
                perf_dict['micro'].append(float(line[i_micro:i_micro+8]))
                perf_dict['time'].append(i_time)
    
    if is_best:
        perf = max(perf_dict["micro"])
        idx_ = perf_dict["micro"].index(perf)
        best_f1b = perf_dict["better"][idx_]
        best_f1w = perf_dict['worse'][idx_]
        best_f1n = perf_dict['none'][idx_]
        best_perf_time_str = perf_dict["time"][idx_]
        time_obj = datetime.datetime.strptime(best_perf_time_str, '%m/%d/%Y %H:%M:%S')
    else:
        raise NotImplementedError("xx")
        perf = perf_dict[moi][-1]
        last_perf_time_str = perf_dict["time"][-1]
        time_obj = datetime.datetime.strptime(last_perf_time_str, '%m/%d/%Y %H:%M:%S')

    perf_time = time_obj - starttime_obj
    
    return perf, best_f1b, best_f1w, best_f1n, perf_time

def traverse_batchratio():
    select_range = product(range(1,6), range(1,6))
    results = []
    for cpc, absa in select_range:
        perf, time = parse("log/grid_ratio_{}_{}.log".format(cpc, absa), "micro", False)
        results.append((cpc, absa, perf, time))
    
    dump_pickle("plot/grid_result.pkl", results)


def traverse_DG():
    test_range = product([True, False], [True, False])
    results = []
    exp_id_base = "DG_{}_{}"
    for do_g, do_d in test_range:
        print(do_g, do_d)

        exp_id = exp_id_base.format("D" if do_d else "UnD", "G" if do_g else "UnG")
        perf = parse_all("log/{}.log".format(exp_id), True)
        results.append((exp_id, *perf))
    
    dump_pickle("plot/dg_results.pkl", results)


if __name__ == "__main__":
    traverse_DG()
    


                

