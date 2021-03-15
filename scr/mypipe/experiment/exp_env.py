import os
import shutil


def make_env(config, save_scr=False, copy_dirs=None):
    if copy_dirs is not None:
        try:
            _copy_dirs(config, copy_dirs)
        except:
            print("dirs have been exist")

    dirs = [
        config.INPUT,
        config.OUTPUT,
        config.SUBMISSION,
        config.FEATURE,
        config.NOTEBOOKS,
        config.EXP,
        config.PREDS,
        config.COLS,
        config.TRAINED,
        config.REPORTS
    ]

    for v in dirs:
        if not os.path.isdir(v):
            print(f"making {v}")
            os.makedirs(v)

    if save_scr:
        shutil.copy(f"{config.EXP_NAME}.py", config.EXP)  # save scr


def _copy_dirs(config, dirs):
    exp_dir = config.OUTPUT + f"/{dirs['EXP_NAME']}"

    if "FEATURE" in dirs["DIRS"]:
        source = os.path.join(exp_dir, "feature")
        copy_to = config.FEATURE
        shutil.copytree(source, copy_to)
        print(f"features dir is copied from {dirs['EXP_NAME']}")

    if "COLS" in dirs["DIRS"]:
        source = os.path.join(exp_dir, "cols")
        copy_to = config.COLS
        shutil.copytree(source, copy_to)
        print(f"cols dir is copied from {dirs['EXP_NAME']}")

    if "PREDS" in dirs["DIRS"]:
        source = os.path.join(exp_dir, "preds")
        copy_to = config.PREDS
        shutil.copytree(source, copy_to)
        print(f"preds dir is copied from {dirs['EXP_NAME']}")
