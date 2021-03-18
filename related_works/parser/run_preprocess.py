from parsers import semeval_15_16
import pickle

def dump_pickle(path, obj):
    """ dump object to pickle file """
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)

if __name__ == "__main__":
    path = "./data/SemEval2016/absa_train.xml"
    res = semeval_15_16(path)
    dump_pickle("./data/SemEval2016/semeval16_res.pkl", res)
