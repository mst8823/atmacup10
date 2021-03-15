from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier

from scr.mypipe.models.base import BaseModel


class MyExTreesClassifier(BaseModel):

    def build_model(self):
        model = ExtraTreesClassifier(**self.params)
        return model

    def predict(self, x):
        preds = self.model.predict_proba(x)[:, 1]
        return preds


class MyExTreesRegressor(BaseModel):

    def build_model(self):
        model = ExtraTreesRegressor(**self.params)
        return model

    def predict(self, x):
        preds = self.model.predict(x)
        return preds
