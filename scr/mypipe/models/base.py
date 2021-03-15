from abc import ABCMeta
from ..utils import Util


class BaseModel(metaclass=ABCMeta):
    def __init__(self, name=None, params=None, fit_params=None):
        self.name = name
        self.params = params
        self.fit_params = fit_params

        self.model = None
        self.scaler = None

    def build_model(self, **kwargs):
        raise NotImplementedError

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.model = self.build_model()  # build model
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        preds = self.model.predict(x)
        return preds

    def save_model(self):
        model_name = f"{self.name}.pkl"
        Util.dump(self.model, model_name)  # save model

    def load_model(self):
        model_name = f"{self.name}.pkl"
        self.model = Util.load(model_name)
