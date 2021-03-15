class BaseBlock(object):
    def fit(self, input_df, y=None):
        raise NotImplementedError

    def transform(self, input_df):
        raise NotImplementedError

    def fit_transform(self, input_df):
        self.fit(input_df)
        return self.transform(input_df)
