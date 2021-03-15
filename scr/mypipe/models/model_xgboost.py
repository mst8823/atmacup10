from xgboost import XGBModel
from scr.mypipe.models.base import BaseModel


class MyXGBModel(BaseModel):

    def build_model(self):
        model = XGBModel(**self.params)
        return model

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.model = self.build_model()
        self.model.fit(tr_x, tr_y,
                       eval_set=[(va_x, va_y)],
                       **self.fit_params)
