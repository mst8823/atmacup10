import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from scr.mypipe.utils import Util
from scr.mypipe.models.base import BaseModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class MyMLPModel(BaseModel):

    def build_model(self, input_dim, output_dim=1):
        model = self.params["model"](input_dim, output_dim)
        return model

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, model_name=None):

        if self.params["scale"]:
            scaler = StandardScaler()
            scaler.fit(tr_x)
            tr_x = scaler.transform(tr_x)
            va_x = scaler.transform(va_x)
            self.scaler = scaler

        if self.params["multiple_target"]:
            tr_y = tf.keras.utils.to_categorical(tr_y)
            va_y = tf.keras.utils.to_categorical(va_y)

        output_dim = tr_y.shape[1] if np.ndim(tr_y) > 1 else 1
        self.model = self.build_model(input_dim=tr_x.shape[1], output_dim=output_dim)

        self.model.fit(tr_x, tr_y,
                       validation_data=(va_x, va_y),
                       **self.fit_params)

    def predict(self, x):
        if self.params["scale"]:
            x = self.scaler.transform(x)
        preds = self.model.predict(x)
        return preds

    def save_model(self):
        model_path = f'{self.name}.h5'
        scaler_path = os.path.join(f'{self.name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self):
        model_path = os.path.join(f'{self.name}.h5')
        scaler_path = os.path.join(f'{self.name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)
