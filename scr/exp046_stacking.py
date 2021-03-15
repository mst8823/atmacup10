import os
import warnings
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

sys.path.append("../")

from scr.mypipe.config import Config
from scr.mypipe.experiment import exp_env
from scr.mypipe.experiment.runner import Runner

from scr.mypipe.models.model_ridge import MyRidgeModel

from scr.mypipe.models.utils import GroupKFold
from scr.mypipe.utils import Util

# ---------------------------------------------------------------------- #
# TODO: stacking
config = Config(EXP_NAME="exp046", TARGET="likes")
exp_env.make_env(config)
os.environ["PYTHONHASHSEED"] = "0"


# ------------------------------------------------------------------------------- #
# ------ Train & predict ---------


# make fold
def make_skf(train_x, train_y, n_splits, random_state=2020):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    s = 10
    _y = pd.cut(train_y, s, labels=range(s))
    return list(skf.split(train_x, _y))


def make_gkf(train_x, train_y, n_splits, random_state=2020):
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    group = pd.read_csv(config.INPUT + "/train.csv")["art_series_id"]
    return list(gkf.split(train_x, train_y, group=group))


# plot result
def result_plot(train_y, oof):
    name = "result"
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.distplot(train_y, label='train_y', color='orange')
    sns.distplot(oof, label='oof')
    ax.legend()
    ax.grid()
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(config.REPORTS, f'{name}.png'), dpi=120)  # save figure
    plt.show()


# create submission
def create_submission(preds):
    sample_sub = pd.read_csv(os.path.join(config.INPUT, "atmacup10__sample_submission.csv"))
    post_preds = [0 if x < 0 else x for x in preds]
    sample_sub["likes"] = post_preds
    sample_sub.to_csv(os.path.join(config.SUBMISSION, f'{config.EXP_NAME}.csv'), index=False)


def main():
    warnings.filterwarnings("ignore")
    train = pd.read_csv(os.path.join(config.INPUT, "train.csv"))

    exp_lst = ["exp015", "exp016", "exp023", "exp024", "exp025", "exp026", "exp027", "exp029", "exp035", "exp041", "exp043", "exp044"]
    train_x, test_x = pd.DataFrame(), pd.DataFrame()
    for exp in exp_lst:
        train_x[exp] = Util.load(f"../output/{exp}/preds/oof_likes.pkl")
        test_x[exp] = Util.load(f"../output/{exp}/preds/preds_likes.pkl")

    train_x = np.log1p(train_x)
    test_x = np.log1p(test_x)
    train_y = train[config.TARGET]

    # model
    model = MyRidgeModel

    # set run params
    run_params = {
        "metrics": mean_squared_error,
        "cv": make_gkf,
        "feature_select_method": None,
        "feature_select_fold": None,
        "feature_select_num": None,
        "folds": 10,
        "seeds": list(range(10000, 10015)),
    }

    # set model params
    model_params = {"random_state": 2021}
    # fit params
    fit_params = {}

    # features
    features = {
        "train_x": train_x,
        "test_x": test_x,
        "train_y": np.log1p(train_y)
    }

    # run
    config.RUN_NAME = f"_{config.TARGET}"
    runner = Runner(config=config,
                    run_params=run_params,
                    model_params=model_params,
                    fit_params=fit_params,
                    model=model,
                    features=features,
                    use_mlflow=False
                    )
    runner.run_train_cv()
    runner.run_predict_cv()

    # make submission
    create_submission(preds=np.expm1(runner.preds))

    # plot result
    result_plot(train_y=np.log1p(train_y), oof=runner.oof)


# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
