import re
import pandas as pd
import hashlib

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from scr.mypipe.features.base import BaseBlock


# ------------------------------------------------------------------- #

class BasicTextFeatureTransformerEngine(BaseBlock):
    def __init__(self, text_columns, cleansing_hero=None, name=""):
        """

        :param text_columns:

            ["col1", "col2", ...,]

        :param cleansing_hero:

            def cleansing_hero(input_df, text_col):
                custom_pipeline = [
                    preprocessing.fillna,
                    preprocessing.remove_urls,
                    preprocessing.remove_html_tags,
                    preprocessing.lowercase,
                    preprocessing.remove_digits,
                    preprocessing.remove_punctuation,
                    preprocessing.remove_diacritics,
                    preprocessing.remove_stopwords,
                    preprocessing.remove_whitespace,
                    preprocessing.stem
                ]
                texts = hero.clean(input_df[text_col], custom_pipeline)
                return texts

        :param name:
        """
        self.text_columns = text_columns
        self.cleansing_hero = cleansing_hero
        self.name = name

        self.df = None

    def fit(self, input_df, y=None):
        _df = pd.DataFrame()
        _df[self.text_columns] = input_df[self.text_columns].astype(str).fillna('missing')
        _df_lst = []
        for c in self.text_columns:
            if self.cleansing_hero is not None:
                _df[c] = self.cleansing_hero(_df, c)
            _df_lst.append(self._get_features(_df, c))
        output_df = pd.concat(_df_lst, axis=1)

        self.df = output_df

    def transform(self, input_df):
        return self.df

    def _get_features(self, dataframe, column):
        output_df = pd.DataFrame()
        output_df[column + self.name + '_num_chars'] = dataframe[column].apply(len)
        output_df[column + self.name + '_num_punctuation'] = dataframe[column].apply(
            lambda x: sum(x.count(w) for w in '.,;:'))
        output_df[column + self.name + '_num_words'] = dataframe[column].apply(lambda x: len(x.split()))
        output_df[column + self.name + '_num_unique_words'] = dataframe[column].apply(
            lambda x: len(set(w for w in x.split())))
        output_df[column + self.name + '_words_vs_unique'] = \
            output_df[column + self.name + '_num_unique_words'] / output_df[column + self.name + '_num_words']
        output_df[column + self.name + '_words_vs_chars'] = \
            output_df[column + self.name + '_num_words'] / output_df[column + self.name + '_num_chars']
        return output_df


class BasicCountLangEngine(BaseBlock):
    def __init__(self,
                 text_columns,
                 cleansing_hero=None,
                 name="_lang"
                 ):
        self.text_columns = text_columns
        self.cleansing_hero = cleansing_hero
        self.name = name
        self.df = None

    def fit(self, input_df, y=None):
        output_df = pd.DataFrame()
        output_df[self.text_columns] = input_df[self.text_columns].astype(str).fillna('missing')
        for c in self.text_columns:
            if self.cleansing_hero is not None:
                output_df[c] = self.cleansing_hero(output_df, c)

            output_df = self._get_features(output_df, c)
        self.df = output_df

    def transform(self, input_df):
        return self.df

    def _get_features(self, dataframe, column):
        output_df = pd.DataFrame()
        output_df[column + self.name + '_num_chars'] = dataframe[column].apply(len)
        output_df[column + self.name + '_num_words'] = dataframe[column].apply(lambda x: len(x.split()))
        output_df[column + self.name + '_num_unique_words'] = dataframe[column].apply(
            lambda x: len(set(w for w in x.split())))
        output_df[column + self.name + '_num_en_chars'] = dataframe[column].apply(lambda x: self._count_roma_word(x))
        output_df[column + self.name + '_num_ja_chars'] = dataframe[column].apply(
            lambda x: self._count_japanese_word(x))
        output_df[column + self.name + '_num_ja_chars_vs_chars'] \
            = output_df[column + self.name + '_num_ja_chars'] / (output_df[column + self.name + '_num_chars'] + 1)
        output_df[column + self.name + '_num_en_chars_vs_chars'] \
            = output_df[column + self.name + '_num_en_chars'] / (output_df[column + self.name + '_num_chars'] + 1)

        return output_df

    @staticmethod
    def _count_japanese_word(strings):
        p = re.compile(
            "[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]|[\uD840-\uD87F]|[\uDC00-\uDFFF]|[ぁ-んァ-ヶ]|[Ａ-Ｚ]|[ｦ-ﾟ]|[ａ-ｚ]")
        count_ja_words = len(p.findall(strings))
        return count_ja_words

    @staticmethod
    def _count_roma_word(strings):
        count_en_word = len(re.findall("[a-zA-Z]", strings))
        return count_en_word


class TextVectorizer(BaseBlock):

    def __init__(self,
                 text_columns,
                 cleansing_hero=None,
                 vectorizer=CountVectorizer(),
                 transformer=TruncatedSVD(n_components=128),
                 transformer2=None,
                 name='',
                 ):
        self.text_columns = text_columns
        self.n_components = transformer.n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.transformer2 = transformer2
        self.name = name
        self.cleansing_hero = cleansing_hero

        self.df = None

    def fit(self, input_df, y=None):
        output_df = pd.DataFrame()
        output_df[self.text_columns] = input_df[self.text_columns].astype(str).fillna('missing')
        features = []
        for c in self.text_columns:
            if self.cleansing_hero is not None:
                output_df[c] = self.cleansing_hero(output_df, c)

            sentence = self.vectorizer.fit_transform(output_df[c])
            feature = self.transformer.fit_transform(sentence)

            if self.transformer2 is not None:
                feature = self.transformer2.fit_transform(feature)

            num_p = feature.shape[1]
            feature = pd.DataFrame(feature, columns=[f"{c}_{self.name}{num_p}" + f'={i:03}' for i in range(num_p)])
            features.append(feature)
        output_df = pd.concat(features, axis=1)
        self.df = output_df

    def transform(self, input_df):
        return self.df


class Doc2VecFeatureTransformer(BaseBlock):

    def __init__(self, text_columns, cleansing_hero=None, params=None, name='doc2vec'):
        self.text_columns = text_columns
        self.cleansing_hero = cleansing_hero
        self.name = name
        self.params = params
        self.df = None

    def fit(self, input_df, y=None):
        dfs = []
        for c in self.text_columns:
            texts = input_df[c].astype(str)
            if self.cleansing_hero is not None:
                texts = self.cleansing_hero(input_df, c)
            texts = [text.split() for text in texts]

            corpus = [TaggedDocument(words=text, tags=[i]) for i, text in enumerate(texts)]
            self.params["documents"] = corpus
            model = Doc2Vec(**self.params, hashfxn=hashfxn)

            result = np.array([model.infer_vector(text) for text in texts])
            output_df = pd.DataFrame(result)
            output_df.columns = [f'{c}_{self.name}:{i:03}' for i in range(result.shape[1])]
            dfs.append(output_df)
        output_df = pd.concat(dfs, axis=1)
        self.df = output_df

    def transform(self, dataframe):
        return self.df

def hashfxn(x):
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)
# ------------------------------------------------------------------- #
