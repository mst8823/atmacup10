import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import inspection
from sklearn.metrics import make_scorer
import mlflow
from ..utils import Util, Logger


class Runner:

    def __init__(self, config, run_params, model_params, fit_params, model, features, use_mlflow=False):
        """
        :param config: config class
        :param run_params: {"metrics": optimized_f1,
                            "cv": make_skf,
                            "feature_select_method": "tree_importance",
                            "feature_select_fold": 5,
                            "feature_select_num": None,
                            "folds": 5,
                            "seeds": [0]}

        :param model_params: {"n_estimators": 100,
                              "objective": 'binary',
                              "learning_rate": 0.01,
                              "num_leaves": 31,
                              "random_state": 2021,
                              "n_jobs": -1,
                              "importance_type": "gain",
                              'colsample_bytree': .5,
                              "reg_lambda": 5}

        :param fit_params: {"early_stopping_rounds":10, "verbose":1}

        :param model: Wrapper Model class
        :param features: {"train_x":train_x, "train_y":train_y, "test_x":test_x}
        """
        self.run_params = run_params
        self.model_params = model_params
        self.fit_params = fit_params
        self.model = model
        self.features = features
        self.config = config
        self.exp_name = config.EXP_NAME
        self.run_name = config.RUN_NAME
        self.exp_run_name = config.EXP_NAME + config.RUN_NAME
        self.folds = self.run_params["folds"]
        self.cv = self.run_params["cv"]
        self.metrics = self.run_params["metrics"]
        self.seeds = run_params["seeds"] if run_params["seeds"] is not None else [2021]
        self.logger = Logger(config.REPORTS)
        self.oof = None
        self.preds = None
        self.use_mlflow = use_mlflow

        # save log
        self.logger.info(f"exp name : {config.EXP_NAME}")
        self.logger.info(f"metrics : {run_params['metrics'].__name__}")
        self.logger.info(f"cv : {run_params['cv'].__name__}")
        self.logger.info(f"model : {model.__name__}")
        self.logger.info(f"feature select method : {run_params['feature_select_method']}")
        self.logger.info(f"feature select fold : {run_params['feature_select_fold']}")
        self.logger.info(f"feature select num : {run_params['feature_select_num']}")
        self.logger.info(f"folds: {run_params['folds']}")
        self.logger.info(f"seeds : {run_params['seeds']}")
        self.logger.info(f"model params : {model_params}")
        self.logger.info(f"fit params: {fit_params}")

        # save mlflow
        if self.use_mlflow:
            mlflow.set_experiment(config.EXP_NAME)
            mlflow.start_run(run_name=self.run_name)
            mlflow.log_param('model_name', model.__name__)
            mlflow.log_param('metrics', run_params['metrics'].__name__)
            mlflow.log_param('cv', run_params['cv'].__name__)
            mlflow.log_params(model_params)
            mlflow.log_params(fit_params)


    def build_model(self, seed, i_fold):
        model_name = f"{self.config.TRAINED}/SEED{seed}_FOLD{i_fold}{self.run_name}"

        if "random_state" in self.model_params:  # change seed
            self.model_params["random_state"] = seed

        model = self.model(name=model_name, params=self.model_params, fit_params=self.fit_params)
        return model

    def get_score(self, y_true, y_pred):
        if self.metrics is not None:
            score = self.metrics(y_true, y_pred)
        else:
            raise NotImplementedError
        return score

    def train_fold(self, seed, i_fold):

        train_x = self.load_x_train()
        train_y = self.load_y_train()
        tr_idx, va_idx = self.load_index_fold(seed, i_fold)
        tr_x, tr_y = train_x.values[tr_idx], train_y.values[tr_idx]
        va_x, va_y = train_x.values[va_idx], train_y.values[va_idx]

        model = self.build_model(seed, i_fold)
        model.fit(tr_x, tr_y, va_x, va_y)

        va_pred = model.predict(va_x)
        score = self.get_score(va_y, va_pred)
        self.logger.info(f"{self.exp_run_name} - SEED:{seed}, FOLD:{i_fold} >>> {score:.4f}")

        if self.use_mlflow:
            mlflow.log_metric(f'seed{seed}-fold{i_fold}', score)

        return model, va_idx, va_pred, score

    def run_train_cv(self):
        self.logger.info(f'{self.exp_run_name} - start training cv')
        train_y = self.load_y_train()  # y true

        preds_seeds = []
        for seed in self.seeds:
            preds = []
            va_idxes = []
            scores = []

            for i_fold in range(self.folds):
                model, va_idx, va_pred, score = self.train_fold(seed, i_fold)
                model.save_model()  # save model
                va_idxes.append(va_idx)
                scores.append(score)
                preds.append(va_pred)

            # sort as default
            va_idxes = np.concatenate(va_idxes)
            order = np.argsort(va_idxes)
            preds = np.concatenate(preds, axis=0)
            preds = preds[order]
            preds_seeds.append(preds)
            score = self.get_score(train_y, preds)
            self.logger.info(f'{self.exp_run_name} - SEED:{seed} - score: {score:.4f}')

        oof = np.mean(preds_seeds, axis=0)
        score = self.get_score(train_y, oof)
        Util.dump(oof, f'{self.config.PREDS}/oof{self.run_name}.pkl')
        self.logger.info(f'{self.exp_run_name} - end training cv - score: {score:.4f}')

        if self.use_mlflow:
            mlflow.log_metric('overall_score', score)
            mlflow.end_run()

        self.oof = oof

    def run_predict_cv(self) -> None:
        self.logger.info(f'{self.exp_run_name} - start prediction cv')
        test_x = self.load_x_test()

        preds_seeds = []
        for seed in self.seeds:
            preds = []
            for i_fold in range(self.folds):
                self.logger.info(f"{self.exp_run_name} >>> SEED:{seed}, FOLD:{i_fold}")
                model = self.build_model(seed, i_fold)
                model.load_model()
                pred = model.predict(test_x.values)
                preds.append(pred)
            preds = np.mean(preds, axis=0)
            preds_seeds.append(preds)

        preds = np.mean(preds_seeds, axis=0)
        Util.dump(preds, f'{self.config.PREDS}/preds{self.run_name}.pkl')
        self.logger.info(f'{self.exp_run_name} - end prediction cv')
        self.preds = preds

    def load_index_fold(self, seed, i_fold):
        train_y = self.load_y_train()
        train_x = self.load_x_train()
        fold = self.cv(train_x, train_y, self.folds, seed)
        return fold[i_fold]

    def load_x_train(self):
        file_path = os.path.join(self.config.COLS, f"cols{self.run_name}.pkl")
        if os.path.isfile(file_path):
            cols = Util.load(file_path)
        else:
            cols = self.get_features_name()
            Util.dump(cols, file_path)
        num = self.run_params["feature_select_num"]
        num = num if num is not None else len(cols)
        cols = cols[:num]
        return self.features["train_x"][cols]

    def load_y_train(self):
        return self.features["train_y"]

    def load_x_test(self):
        file_path = os.path.join(self.config.COLS, f"cols{self.run_name}.pkl")
        if os.path.isfile(file_path):
            cols = Util.load(file_path)
        else:
            cols = self.get_features_name()
            Util.dump(cols, file_path)
        num = self.run_params["feature_select_num"]
        num = num if num is not None else len(cols)
        cols = cols[:num]
        return self.features["test_x"][cols]

    def get_features_name(self):
        if self.run_params["feature_select_method"] is None:
            train_x = self.features["train_x"]
            cols = train_x.columns.tolist()

        elif self.run_params["feature_select_method"] == "tree_importance":
            imp_df = self.tree_importance()
            cols = imp_df["column"].to_list()  # get selected col names

        elif self.run_params["feature_select_method"] == "permutation_importance":
            imp_df = self.permutation_importance()
            cols = imp_df["column"].to_list()  # get selected col names

        else:
            raise NotImplementedError

        return cols

    def tree_importance(self):
        """
        get GBDT feature importance
        :return: importance df
        """
        name = "tree_importance"
        train_x = self.features["train_x"]
        train_y = self.features["train_y"]

        feature_importance_df = pd.DataFrame()
        fold_idx = self.cv(train_x, train_y, self.run_params["feature_select_fold"])
        for i, (tr_idx, va_idx) in enumerate(fold_idx):
            print(f"fold {i} >>>>>")
            tr_x, va_x = train_x.values[tr_idx], train_x.values[va_idx]
            tr_y, va_y = train_y.values[tr_idx], train_y.values[va_idx]

            model = self.build_model(seed=2021, i_fold=i)
            model.fit(tr_x, tr_y, va_x, va_y)

            _df = pd.DataFrame()
            _df['feature_importance'] = model.model.feature_importances_
            _df['column'] = train_x.columns
            _df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

        imp_df = self._get_importance_df(feature_importance_df, name)
        return imp_df

    def permutation_importance(self):
        """
        get permutation importance
        :return: importance df
        """
        name = "permutation_importance"
        get_score = make_scorer(self.metrics)
        train_x = self.features["train_x"]
        train_y = self.features["train_y"]
        feature_importance_df = pd.DataFrame()
        fold_idx = self.cv(train_x, train_y, self.run_params["feature_select_fold"])
        for i, (tr_idx, va_idx) in enumerate(fold_idx):
            print(f"fold {i} >>>>>")
            tr_x, va_x = train_x.values[tr_idx], train_x.values[va_idx]
            tr_y, va_y = train_y.values[tr_idx], train_y.values[va_idx]

            model = self.build_model(seed=2021, i_fold=i)
            model.fit(tr_x, tr_y, va_x, va_y)

            _df = pd.DataFrame()
            result = inspection.permutation_importance(estimator=model.model,
                                                       X=va_x, y=va_y,
                                                       scoring=get_score,
                                                       n_repeats=5,
                                                       n_jobs=-1,
                                                       random_state=2021)
            _df['feature_importance'] = result["importances_mean"]
            _df['column'] = train_x.columns
            _df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

        imp_df = self._get_importance_df(feature_importance_df, name)
        return imp_df

    def _get_importance_df(self, feature_importance_df, name):
        order = feature_importance_df.groupby('column').sum()[['feature_importance']]
        order = order.sort_values('feature_importance', ascending=False).index[:50]
        fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
        sns.boxenplot(data=feature_importance_df,
                      y='column',
                      x='feature_importance',
                      order=order,
                      ax=ax,
                      palette='viridis')
        ax.grid()
        ax.set_title(name)
        fig.tight_layout()
        plt.show()

        imp_df = feature_importance_df.groupby("column", as_index=False).mean()
        imp_df = imp_df.sort_values("feature_importance", ascending=False)
        imp_df = imp_df.query('feature_importance > 0')[["column", "feature_importance"]]  # remove importance = 0

        fig.savefig(os.path.join(self.config.REPORTS, f'{name}{self.run_name}_fig.png'), dpi=120)  # save figure
        imp_df.to_csv(os.path.join(self.config.REPORTS, f'{name}{self.run_name}_df.csv'), index=False)  # save df
        return imp_df
