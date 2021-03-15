import os
import warnings
import sys
import hashlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from langdetect import DetectorFactory, detect
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from geopy.geocoders import Nominatim
import texthero as hero
import nltk
from gensim.models import word2vec

sys.path.append("../")

from scr.mypipe.config import Config
from scr.mypipe.experiment import exp_env
from scr.mypipe.experiment.runner import Runner
from scr.mypipe.features.feature_decorator import cach_feature
from scr.mypipe.features.feature_engine import CountEncodingEngine, OrdinalEncodingEngine, GroupingEngine, \
    TargetEncodingEngine
from scr.mypipe.features.text_engine import BasicTextFeatureTransformerEngine, TextVectorizer, Doc2VecFeatureTransformer

from scr.mypipe.models.model_lgbm import MyLGBMModel
from scr.mypipe.models.utils import GroupKFold

# ---------------------------------------------------------------------- #
# TODO: add stop word (text)
config = Config(EXP_NAME="exp041", TARGET="likes")
exp_env.make_env(config)
DetectorFactory.seed = 0
os.environ["PYTHONHASHSEED"] = "0"

color = pd.read_csv(config.INPUT + "/color.csv")
historical_person = pd.read_csv(config.INPUT + "/historical_person.csv")
maker = pd.read_csv(config.INPUT + "/maker.csv")
material = pd.read_csv(config.INPUT + "/material.csv")
object_collection = pd.read_csv(config.INPUT + "/object_collection.csv")
palette = pd.read_csv(config.INPUT + "/palette.csv")
principal_maker_occupation = pd.read_csv(config.INPUT + "/principal_maker_occupation.csv")
principal_maker = pd.read_csv(config.INPUT + "/principal_maker.csv")
production_place = pd.read_csv(config.INPUT + "/production_place.csv")
technique = pd.read_csv(config.INPUT + "/technique.csv")


# ---------------------------------------------------------------------- #
def hex2rgb(color_code):
    color_code = color_code[1:]
    red = int(color_code[1:3], 16)
    green = int(color_code[3:5], 16)
    blue = int(color_code[5:7], 16)
    return [red, green, blue]


@cach_feature(feature_name="color_agg_features", directory=config.FEATURE)
def get_color_agg_features(input_df):
    tmp = np.array([hex2rgb(x) for x in color["hex"]])
    rgb_df = pd.DataFrame(tmp * (color[["percentage"]] / 100).values, columns=["color_r", "color_g", "color_b"])
    _df = pd.concat([color[["object_id"]], rgb_df], axis=1)
    _df = _df.groupby("object_id").agg(["min", "max", "mean", "median", "sum"])
    _df.columns = [f"agg_color_{cols[1]}_{cols[0]}" for cols in _df.columns]
    output_df = pd.merge(input_df[["object_id"]], _df.reset_index(), on="object_id", how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="production_place_country_features", directory=config.FEATURE)
def get_production_place_country_features(input_df):
    def place2country(address):
        geolocator = Nominatim(user_agent='sample', timeout=200)
        loc = geolocator.geocode(address, language='en')
        coordinates = (loc.latitude, loc.longitude)
        location = geolocator.reverse(coordinates, language='en')
        country = location.raw['address']['country']
        return country

    _production_place = production_place.copy()
    place_list = _production_place['name'].unique()
    country_dict = {}
    for place in tqdm(place_list):
        try:
            country = place2country(place)
            country_dict[place] = country
        except:
            # 国名を取得できない場合はnan
            print(place)
            country_dict[place] = np.nan

    _production_place['country_name'] = _production_place['name'].map(country_dict)
    _production_place["const"] = 1
    _df = pd.pivot_table(_production_place, index="object_id", columns="country_name", values="const").fillna(0)
    output_df = pd.merge(input_df[["object_id"]], _df, on="object_id", how="left").drop("object_id", axis=1)
    return output_df.add_prefix("production_country=")


@cach_feature(feature_name="technique_features", directory=config.FEATURE)
def get_technique_features(input_df):
    technique["const"] = 1
    _df = pd.pivot_table(technique, index="object_id", columns="name", values="const").fillna(0).add_prefix(
        "technique=")
    _df.columns = [x.replace(" ", "_") for x in _df.columns]
    output_df = pd.merge(input_df[["object_id"]], _df.reset_index(), on="object_id", how="left")

    def get_pca(df, n_components, name):
        sc_df = StandardScaler().fit_transform(df)
        pca = PCA(n_components=n_components, random_state=2021)
        return pd.DataFrame(pca.fit_transform(sc_df)).rename(columns=lambda x: f"{name}={x:03}")

    pca_df = get_pca(_df, 8, "technique_PCA")
    pca_df["object_id"] = _df.index

    output_df["num_of_technique"] = output_df.drop("object_id", axis=1).sum(axis=1)
    output_df = pd.merge(output_df, pca_df, on="object_id", how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="palette_agg_features", directory=config.FEATURE)
def get_palette_agg_features(input_df):
    _palette = palette.copy()
    _palette["color_r&b"] = _palette["color_r"] + _palette["color_b"]
    _palette["color_r&g"] = _palette["color_r"] + _palette["color_g"]
    _palette["color_b&g"] = _palette["color_b"] + _palette["color_g"]
    _palette["color_r&b&g"] = _palette["color_r"] + _palette["color_b"] + _palette["color_g"]

    _df = _palette.groupby("object_id").agg(["min", "max", "mean", "median", "sum", "std"])
    _df.columns = [f"agg_palette_{cols[1]}_{cols[0]}" for cols in _df.columns]
    output_df = pd.merge(input_df[["object_id"]], _df.reset_index(), on="object_id", how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="maker_features", directory=config.FEATURE)
def get_maker_features(input_df):
    _df = pd.merge(input_df[["object_id", "principal_maker"]], maker, left_on="principal_maker", right_on="name",
                   how="left")
    _df["date_of_birth"] = [int(x[:4]) for x in _df["date_of_birth"].fillna("0000")]
    _df["date_of_death"] = [int(x[:4]) for x in _df["date_of_death"].fillna("0000")]  # year になおす
    _df[["date_of_birth", "date_of_death"]] = _df[["date_of_birth", "date_of_death"]].replace(0, np.nan)
    _df["duration_birth2death"] = _df["date_of_death"] - _df["date_of_birth"]
    output_df = _df.drop(["object_id", "principal_maker", "name"], axis=1)
    return output_df


@cach_feature(feature_name="palette_features", directory=config.FEATURE)
def get_palette_features(input_df):
    def get_pca(df, n_components, name):
        sc_df = StandardScaler().fit_transform(df)
        pca = PCA(n_components=n_components, random_state=2021)
        return pd.DataFrame(pca.fit_transform(sc_df)).rename(columns=lambda x: f"{name}={x:03}")

    _palette = palette.copy()
    _palette["No"] = np.tile(range(22), len(np.unique(_palette["object_id"])))

    _red = pd.pivot_table(_palette, index="object_id", columns="No", values="color_r")
    _green = pd.pivot_table(_palette, index="object_id", columns="No", values="color_g")
    _blue = pd.pivot_table(_palette, index="object_id", columns="No", values="color_b")
    _all = pd.concat([_red, _green, _blue], axis=1)

    _red_pca = get_pca(_red, 8, "palette_red_PCA")
    _blue_pca = get_pca(_blue, 8, "palette_blue_PCA")
    _green_pca = get_pca(_green, 8, "palette_blue_PCA")
    _all_pca = get_pca(_all, 32, "palette_all_color_PCA")

    _df = pd.concat([
        _red_pca,
        _blue_pca,
        _green_pca,
        _all_pca,
        _blue.reset_index(drop=True).add_prefix("pallete_blue=")
    ], axis=1)
    _df["object_id"] = _red.index
    output_df = pd.merge(input_df[["object_id"]], _df, on="object_id", how="left")

    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="principal_maker_occupation_features", directory=config.FEATURE)
def get_principal_maker_occupation_features(input_df):
    _df = pd.merge(principal_maker, principal_maker_occupation, on="id", how="left")
    output_df = pd.crosstab(index=_df["object_id"], columns=_df["name"]).add_prefix("occupation=")
    output_df["num_of_occupation"] = output_df.sum(axis=1)
    output_df = pd.merge(input_df[["object_id"]], output_df.reset_index(), on="object_id", how="left")

    return output_df.drop("object_id", axis=1)


# ------------------------------------------------------------------------------- #
@cach_feature(feature_name="raw_features", directory=config.FEATURE)
def get_raw_features(input_df):
    cols = (input_df.dtypes[input_df.dtypes != "object"]).index.tolist()
    return input_df[cols].copy()


@cach_feature(feature_name="acquisition_date_feature", directory=config.FEATURE)
def get_acquisition_date_feature(input_df):
    output_df = pd.DataFrame()
    tmp = pd.to_datetime(input_df["acquisition_date"])
    output_df["acquisition_date_year"] = tmp.dt.year
    output_df["acquisition_date_month"] = tmp.dt.month
    return output_df


@cach_feature(feature_name="dating_features", directory=config.FEATURE)
def get_dating_features(input_df):
    output_df = pd.DataFrame()

    output_df["diff_dating_year_late2early"] = input_df["dating_year_late"] - input_df["dating_year_early"]
    output_df["dating_sorting_date:02"] = [str(x)[:2] for x in input_df["dating_sorting_date"].fillna(0)]
    output_df["dating_sorting_date:02"] = output_df["dating_sorting_date:02"].replace("0", np.nan).astype(float)
    return output_df


@cach_feature(feature_name="sub_title_features", directory=config.FEATURE)
def get_sub_title_features(input_df):
    tmp = input_df["sub_title"].replace('h 166mm × w 78/54mm', 'h 166mm × w 66mm').fillna("NA")
    output_df = pd.DataFrame()
    for axis in ['h', 'w', 't', 'd']:
        column_name = f'sub_title={axis}'
        size_info = tmp.str.extract(r'{} (\d*|\d*\.\d*)(cm|mm)'.format(axis))  # 正規表現を使ってサイズを抽出
        size_info = size_info.rename(columns={0: column_name, 1: 'unit'})

        size_info[column_name] = size_info[column_name].replace('', np.nan).astype(float)  # dtypeがobjectになってるのでfloatに直す
        size_info[column_name] = size_info.apply(
            lambda row: row[column_name] * 10 if row['unit'] == 'cm' else row[column_name], axis=1)  # 単位をmmに統一する
        output_df[column_name] = size_info[column_name]

    output_df["sub_title_h*w"] = output_df[["sub_title=h", "sub_title=w"]].prod(axis=1)
    output_df["sub_title_h*t"] = output_df[["sub_title=h", "sub_title=t"]].prod(axis=1)
    output_df["sub_title_h*d"] = output_df[["sub_title=h", "sub_title=t"]].prod(axis=1)
    output_df["sub_title_w*d"] = output_df[["sub_title=w", "sub_title=d"]].prod(axis=1)
    output_df["sub_title_w*t"] = output_df[["sub_title=w", "sub_title=t"]].prod(axis=1)
    output_df["sub_title_h*w*t*d"] = output_df.prod(axis=1)
    return output_df


@cach_feature(feature_name="ce_features", directory=config.FEATURE)
def get_ce_features(input_df):
    _input_df = pd.concat([
        input_df,
    ], axis=1)
    cols = [
        "art_series_id",
        "title",
        "sub_title",
        "description",
        "long_title",
        "principal_maker",
        "principal_or_first_maker",
        "copyright_holder",
        "more_title",
        "acquisition_method",
        "acquisition_credit_line",
        "place_of_birth",
        "place_of_death",
        "nationality",
    ]
    encoder = CountEncodingEngine(cols=cols)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@cach_feature(feature_name="oe_features", directory=config.FEATURE)
def get_oe_features(input_df):
    _input_df = pd.concat([
        input_df,
    ], axis=1)
    cols = [
        "acquisition_method",
    ]
    encoder = OrdinalEncodingEngine(cols=cols)
    output_df = encoder.fit_transform(_input_df)
    return output_df


@cach_feature(feature_name="agg_nunique_principal_maker_features", directory=config.FEATURE)
def get_agg_nunique_principal_maker_features(input_df):
    group_key = "principal_maker"
    group_values = [
        "art_series_id",
        "title",
        "principal_or_first_maker",
        "copyright_holder",
        "dating_sorting_date"
    ]
    agg_methods = [pd.Series.nunique]
    encoder = GroupingEngine(group_key=group_key, group_values=group_values, agg_methods=agg_methods)

    output_df = encoder.fit_transform(input_df)
    return output_df


@cach_feature(feature_name="agg_principal_maker_features", directory=config.FEATURE)
def get_agg_principal_maker_features(input_df):
    _input_df = pd.concat([input_df,
                           get_sub_title_features(input_df),
                           get_cross_num_features(input_df),
                           get_dating_features(input_df)
                           ], axis=1)

    def max_min(x):
        return x.max() - x.min()

    group_key = "principal_maker"
    group_values = [
        "diff_dating_year_late2early",
        "dating_sorting_date",

        "sub_title_h*w*t*d",
        "sub_title_h*w",

        "agg_palette_sum_color_b",
        "agg_palette_sum_color_g",
        "agg_palette_sum_color_r",
        "agg_palette_sum_color_r&b",
        "agg_palette_sum_color_r&g",
        "agg_palette_sum_color_b&g",
        "agg_palette_sum_color_r&b&g",

        # "prod_palletSumColorB_SubTitleH*W*T*D",
        # "prod_palletSumColorG_SubTitleH*W*T*D",
        # "prod_palletSumColorR_SubTitleH*W*T*D",
        # "prod_palletSumColorB_SubTitleH*W",
        # "prod_palletSumColorG_SubTitleH*W",
        # "prod_palletSumColorR_SubTitleH*W",
        # "prod_palletSumColorRB_SubTitleH*W*T*D",
        # "prod_palletSumColorRG_SubTitleH*W*T*D",
        # "prod_palletSumColorBG_SubTitleH*W*T*D",
        # "prod_palletSumColorRBG_SubTitleH*W*T*D",

        "ratio_palletSumColorB_SubTitleH*W*T*D",
        "ratio_palletSumColorG_SubTitleH*W*T*D",
        "ratio_palletSumColorR_SubTitleH*W*T*D",

        "ratio_datingPeriod_durationSurvival"

    ]
    agg_methods = ["min", "max", "mean", max_min, "sum", "std", "z-score"]
    encoder = GroupingEngine(group_key=group_key, group_values=group_values, agg_methods=agg_methods)

    output_df = encoder.fit_transform(_input_df)
    return output_df


@cach_feature(feature_name="agg_art_series_id_features", directory=config.FEATURE)
def get_agg_art_series_id_features(input_df):
    _input_df = pd.concat([input_df,
                           get_sub_title_features(input_df),
                           get_cross_num_features(input_df),
                           get_dating_features(input_df)
                           ], axis=1)

    def max_min(x):
        return x.max() - x.min()

    group_key = "art_series_id"
    group_values = [
        "diff_dating_year_late2early",
        "dating_sorting_date",
        "sub_title_h*w*t*d",
        "sub_title_h*w",
        "agg_palette_sum_color_b",
        "agg_palette_sum_color_g",
        "agg_palette_sum_color_r",
        "agg_palette_sum_color_r&b",
        "agg_palette_sum_color_r&g",
        "agg_palette_sum_color_b&g",
        "agg_palette_sum_color_r&b&g",
        # "prod_palletSumColorB_SubTitleH*W*T*D",
        # "prod_palletSumColorG_SubTitleH*W*T*D",
        # "prod_palletSumColorR_SubTitleH*W*T*D",
        # "prod_palletSumColorRB_SubTitleH*W*T*D",
        # "prod_palletSumColorRG_SubTitleH*W*T*D",
        # "prod_palletSumColorBG_SubTitleH*W*T*D",
        # "prod_palletSumColorRBG_SubTitleH*W*T*D",
        "ratio_palletSumColorB_SubTitleH*W*T*D",
        "ratio_palletSumColorG_SubTitleH*W*T*D",
        "ratio_palletSumColorR_SubTitleH*W*T*D",
        "ratio_datingPeriod_durationSurvival"
    ]
    agg_methods = ["min", "max", "mean", max_min, "sum", "std", "z-score"]
    encoder = GroupingEngine(group_key=group_key, group_values=group_values, agg_methods=agg_methods)

    output_df = encoder.fit_transform(_input_df)
    return output_df


@cach_feature(feature_name="agg_dating_sorting_date_features", directory=config.FEATURE)
def get_agg_dating_sorting_date_features(input_df):
    _input_df = pd.concat([input_df,
                           get_sub_title_features(input_df),
                           get_cross_num_features(input_df),
                           get_dating_features(input_df)
                           ], axis=1)

    def max_min(x):
        return x.max() - x.min()

    group_key = "dating_sorting_date"
    group_values = [
        "diff_dating_year_late2early",
        "sub_title_h*w*t*d",
        "sub_title_h*w",
        "agg_palette_sum_color_b",
        "agg_palette_sum_color_g",
        "agg_palette_sum_color_r",
        # "prod_palletSumColorB_SubTitleH*W*T*D",
        # "prod_palletSumColorG_SubTitleH*W*T*D",
        # "prod_palletSumColorR_SubTitleH*W*T*D",
        "ratio_datingPeriod_durationSurvival",

    ]
    agg_methods = ["min", "max", "mean", max_min, "sum", "std", "z-score"]
    encoder = GroupingEngine(group_key=group_key, group_values=group_values, agg_methods=agg_methods)

    output_df = encoder.fit_transform(_input_df)
    return output_df


@cach_feature(feature_name="cross_num_features", directory=config.FEATURE)
def get_cross_num_features(input_df):
    _input_df = pd.concat([
        get_palette_agg_features(input_df),
        get_sub_title_features(input_df),
        get_dating_features(input_df),
        get_maker_features(input_df)
    ], axis=1)

    output_df = pd.DataFrame()
    output_df["prod_palletSumColorB_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_b"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)
    output_df["prod_palletSumColorG_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_g"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)
    output_df["prod_palletSumColorR_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_r"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)

    output_df["ratio_palletSumColorB_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_b"] / (
            _input_df["sub_title_h*w*t*d"] + 1e-2) * 1e+7
    output_df["ratio_palletSumColorG_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_g"] / (
            _input_df["sub_title_h*w*t*d"] + 1e-2) * 1e+7
    output_df["ratio_palletSumColorR_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_r"] / (
            _input_df["sub_title_h*w*t*d"] + 1e-2) * 1e+7

    output_df["ratio_palletSumColorB_SubTitleH*W"] = _input_df["agg_palette_sum_color_b"] / (
            _input_df["sub_title_h*w"] + 1e-2) * 1e+7
    output_df["ratio_palletSumColorG_SubTitleH*W"] = _input_df["agg_palette_sum_color_g"] / (
            _input_df["sub_title_h*w"] + 1e-2) * 1e+7
    output_df["ratio_palletSumColorR_SubTitleH*W"] = _input_df["agg_palette_sum_color_r"] / (
            _input_df["sub_title_h*w"] + 1e-2) * 1e+7

    output_df["prod_palletSumColorRB_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_r&b"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)
    output_df["prod_palletSumColorRG_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_r&g"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)
    output_df["prod_palletSumColorBG_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_b&g"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)
    output_df["prod_palletSumColorRBG_SubTitleH*W*T*D"] = _input_df["agg_palette_sum_color_r&b&g"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)

    output_df["prod_palletMeanColorB_SubTitleH*W*T*D"] = _input_df["agg_palette_mean_color_b"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)
    output_df["prod_palletMeanColorG_SubTitleH*W*T*D"] = _input_df["agg_palette_mean_color_g"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)
    output_df["prod_palletMeanColorR_SubTitleH*W*T*D"] = _input_df["agg_palette_mean_color_r"] * (
            _input_df["sub_title_h*w*t*d"] + 1e-2)
    output_df["prod_palletSumColorB_SubTitleH*W"] = _input_df["agg_palette_sum_color_b"] * (
            _input_df["sub_title_h*w"] + 1e-2)
    output_df["prod_palletSumColorG_SubTitleH*W"] = _input_df["agg_palette_sum_color_g"] * (
            _input_df["sub_title_h*w"] + 1e-2)
    output_df["prod_palletSumColorR_SubTitleH*W"] = _input_df["agg_palette_sum_color_r"] * (
            _input_df["sub_title_h*w"] + 1e-2)
    output_df["prod_palletMeanColorB_SubTitleH*W"] = _input_df["agg_palette_mean_color_b"] * (
            _input_df["sub_title_h*w"] + 1e-2)
    output_df["prod_palletMeanColorG_SubTitleH*W"] = _input_df["agg_palette_mean_color_g"] * (
            _input_df["sub_title_h*w"] + 1e-2)
    output_df["prod_palletMeanColorR_SubTitleH*W"] = _input_df["agg_palette_mean_color_r"] * (
            _input_df["sub_title_h*w"] + 1e-2)
    output_df = np.log1p(output_df)

    output_df["ratio_SubTitleH*W*T*D_datingPeriod"] = _input_df["sub_title_h*w*t*d"] / (
            _input_df["diff_dating_year_late2early"] + 1e-3)
    output_df["ratio_datingPeriod_durationSurvival"] = _input_df["diff_dating_year_late2early"] / (
            _input_df["duration_birth2death"] + 1e-3)
    return output_df


@cach_feature(feature_name="basic_text_features", directory=config.FEATURE)
def get_basic_text_features(input_df):
    _input_df = input_df.copy()
    _input_df["concat_title"] = _input_df["title"] + " " + _input_df["long_title"] + " " + _input_df[
        "more_title"].fillna("")
    encoder = BasicTextFeatureTransformerEngine(
        text_columns=["title", "long_title", "more_title", "description", "principal_maker", "concat_title",
                      "sub_title"])
    output_df = encoder.fit_transform(_input_df)
    return output_df


@cach_feature(feature_name="text_vec_features", directory=config.FEATURE)
def get_text_vec_features(input_df):
    def cleansing_hero(df, text_col):
        custom_stopwords = nltk.corpus.stopwords.words('dutch') + \
                           nltk.corpus.stopwords.words('english') + \
                           nltk.corpus.stopwords.words('french') + \
                           nltk.corpus.stopwords.words('german') + \
                           nltk.corpus.stopwords.words('italian')

        custom_pipeline = [
            hero.preprocessing.fillna,
            hero.preprocessing.lowercase,
            hero.preprocessing.remove_digits,
            hero.preprocessing.remove_punctuation,
            hero.preprocessing.remove_diacritics,
            lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords),
            hero.preprocessing.stem
        ]
        texts = hero.clean(df[text_col], custom_pipeline)
        return texts

    _input_df = input_df.copy()
    _input_df["concat_title"] = _input_df["title"] + " " + _input_df["long_title"] + " " + _input_df[
        "more_title"].fillna("")
    encoder = TextVectorizer(
        text_columns=["title", "long_title", "more_title", "description", "concat_title"],
        vectorizer=TfidfVectorizer(),
        cleansing_hero=cleansing_hero,
        transformer=TruncatedSVD(n_components=256, random_state=2021),
        name="SVD")
    output_df = encoder.fit_transform(_input_df)
    return output_df


def get_te_features(input_df):
    target_col = "likes"
    input_cols = [
        "acquisition_method",
        "dating_sorting_date",
    ]
    train_df = input_df[~input_df["likes"].isnull()]
    encoder = TargetEncodingEngine(target_col=target_col, input_cols=input_cols, fold=None)

    encoder.fit(train_df)
    output_df = encoder.transform(input_df)
    return output_df


def _get_w2v_features(df, params, name):
    """
    df: pd.Serise, index=object_id, values=list
    """

    def hashfxn(x):
        return int(hashlib.md5(str(x).encode()).hexdigest(), 16)

    w2v_model = word2vec.Word2Vec(df.values.tolist(),
                                  hashfxn=hashfxn,
                                  **params)
    tqdm.pandas()
    # 各文章ごとにそれぞれの単語をベクトル表現に直し、平均をとって文章ベクトルにする
    sentence_vectors = df.progress_apply(lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
    sentence_vectors = np.vstack([x for x in sentence_vectors])
    sentence_vector_df = pd.DataFrame(sentence_vectors)
    sentence_vector_df.columns = [f"{name}_w2v={x:03}" for x in range(sentence_vector_df.shape[1])]
    sentence_vector_df.index = df.index
    return sentence_vector_df


@cach_feature(feature_name="w2v_tec_features", directory=config.FEATURE)
def get_w2v_tec_features(input_df):
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = technique.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "technique")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_occupation_features", directory=config.FEATURE)
def get_w2v_occupation_features(input_df):
    df = pd.merge(principal_maker, principal_maker_occupation, on="id", how="left").dropna(subset=["name"])
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = df.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "occupation")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_tec_features", directory=config.FEATURE)
def get_w2v_col_features(input_df):
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = object_collection.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "object_collection")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_tec_features", directory=config.FEATURE)
def get_w2v_material_features(input_df):
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = material.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "material")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_tec_features", directory=config.FEATURE)
def get_w2v_production_place_features(input_df):
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = production_place.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "production_place")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_tec_features", directory=config.FEATURE)
def get_w2v_historical_person_features(input_df):
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = historical_person.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "historical_person")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_tec_col_acq_features", directory=config.FEATURE)
def get_w2v_hist_mat_col_tech_pro_acq_features(input_df):
    _df = input_df[["object_id", "acquisition_method"]].rename(columns={"acquisition_method": "name"}).fillna("")
    df = pd.concat([historical_person, material, object_collection, technique, production_place, _df]).reset_index(
        drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = df.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "hist_mat_col_tech_pro_acq")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_mat_col_tec_pro_features", directory=config.FEATURE)
def get_w2v_hist_mat_col_tec_pro_features(input_df):
    hist_mat_col_tec = pd.concat([historical_person, material, object_collection, technique, production_place],
                                 axis=0).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = hist_mat_col_tec.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "hist_mat_col_tec_pro")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_mat_col_tec_features", directory=config.FEATURE)
def get_w2v_mat_col_tec_features(input_df):
    mat_col_tec = pd.concat([material, object_collection, technique], axis=0).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = mat_col_tec.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "mat_col_tec")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_mat_col_tec_pro_features", directory=config.FEATURE)
def get_w2v_mat_col_tec_pro_features(input_df):
    mat_col_tec = pd.concat([material, object_collection, technique, production_place], axis=0).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = mat_col_tec.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "mat_col_tec_pro")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_col_tec_pro_features", directory=config.FEATURE)
def get_w2v_col_tec_pro_features(input_df):
    mat_col_tec = pd.concat([object_collection, technique, production_place], axis=0).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = mat_col_tec.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "col_tec_pro")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_tec_pro_features", directory=config.FEATURE)
def get_w2v_pro_tec_features(input_df):
    mat_col_tec = pd.concat([production_place, technique], axis=0).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = mat_col_tec.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "tec_pro")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_tec_pro_features", directory=config.FEATURE)
def get_w2v_hist_tec_features(input_df):
    mat_col_tec = pd.concat([historical_person, technique], axis=0).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = mat_col_tec.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "hist_pro")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_mat_tec_pro_features", directory=config.FEATURE)
def get_w2v_mat_pro_tec_features(input_df):
    mat_col_tec = pd.concat([material, production_place, technique], axis=0).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = mat_col_tec.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "mat_tec_pro")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_mat_tec_features", directory=config.FEATURE)
def get_w2v_mat_tec_features(input_df):
    _df = pd.concat([material, technique]).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = _df.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "mat_tec")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_col_tec_features", directory=config.FEATURE)
def get_w2v_col_tec_features(input_df):
    _df = pd.concat([object_collection, technique]).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = _df.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "col_tec")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_hist_col_tec_features", directory=config.FEATURE)
def get_w2v_hist_col_tec_features(input_df):
    _df = pd.concat([historical_person, object_collection, technique]).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = _df.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "hist_col_tec")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_tec_col_acq_features", directory=config.FEATURE)
def get_w2v_tec_col_acq_features(input_df):
    _df = input_df[["object_id", "acquisition_method"]].rename(columns={"acquisition_method": "name"}).fillna("")
    df = pd.concat([technique, object_collection, _df]).reset_index(drop=True)
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = df.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "tec_col_acq")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_dating_year_features", directory=config.FEATURE)
def get_w2v_dating_year_features(input_df):
    _df1 = input_df[["object_id", "dating_year_early"]].rename(columns={"dating_year_early": "name"})
    _df2 = input_df[["object_id", "dating_year_late"]].rename(columns={"dating_year_late": "name"})
    df = pd.concat([_df1, _df2]).reset_index(drop=True).astype(str).fillna("")
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }
    _df = df.groupby("object_id")["name"].apply(list)
    _df = _get_w2v_features(_df, params, "dating_year")
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="w2v_palette_features", directory=config.FEATURE)
def get_w2v_palette_features(input_df):
    _df = palette.sort_values(["object_id", "ratio"]).reset_index(drop=True)
    _df[["color_r", "color_g", "color_b"]] = _df[["color_r", "color_g", "color_b"]].astype(int).astype(str)

    _df_r = _df.groupby("object_id")["color_r"].apply(list)
    _df_b = _df.groupby("object_id")["color_b"].apply(list)
    _df_g = _df.groupby("object_id")["color_g"].apply(list)
    _df_dic = {"color_r": _df_r, "color_b": _df_b, "color_g": _df_g}
    params = {
        "size": 32,
        "min_count": 1,
        "window": 1,
        "iter": 200
    }

    dfs = []
    for k, v in _df_dic.items():
        dfs.append(_get_w2v_features(v, params, k))
    _df = pd.concat(dfs, axis=1)
    output_df = pd.merge(input_df[["object_id"]], _df, left_on="object_id", right_index=True, how="left")
    return output_df.drop("object_id", axis=1)


@cach_feature(feature_name="doc2vec_features", directory=config.FEATURE)
def get_doc2vec_features(input_df):
    def cleansing_hero(df, text_col):
        custom_stopwords = nltk.corpus.stopwords.words('dutch') + \
                           nltk.corpus.stopwords.words('english') + \
                           nltk.corpus.stopwords.words('french') + \
                           nltk.corpus.stopwords.words('german') + \
                           nltk.corpus.stopwords.words('italian')

        custom_pipeline = [
            hero.preprocessing.fillna,
            hero.preprocessing.lowercase,
            hero.preprocessing.remove_digits,
            hero.preprocessing.remove_punctuation,
            hero.preprocessing.remove_diacritics,
            lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords),
            hero.preprocessing.stem
        ]
        texts = hero.clean(df[text_col], custom_pipeline)
        return texts

    _input_df = input_df.copy()
    _input_df["concat_title"] = _input_df["title"] \
                                + " " + _input_df["long_title"] \
                                + " " + _input_df["more_title"].fillna("")
    cols = ["concat_title"]

    params = {
        "vector_size": 64,
        "window": 10,
        "min_count": 1,
        "epochs": 20,
        "seed": 2021

    }
    encoder = Doc2VecFeatureTransformer(text_columns=cols, cleansing_hero=cleansing_hero, params=params)
    output_df = encoder.fit_transform(_input_df)
    return output_df


# ------------------------------------------------------------------------------- #

def merge_data(input_df):
    funcs = [
        get_color_agg_features,
        # get_production_place_country_features,
        get_technique_features,
        get_palette_agg_features,
        get_maker_features,
        # get_palette_features,
        # get_principal_maker_occupation_features
    ]

    # preprocess
    output_df = []
    for func in tqdm(funcs):
        _df = func(input_df)
        output_df.append(_df)
    output_df = pd.concat([input_df] + output_df, axis=1)
    return output_df


# main func for preprocessing
def preprocess(train, test):
    # load data
    input_df = pd.concat([train, test]).reset_index(drop=True)  # use concat data
    input_df = merge_data(input_df)  # merge other data

    # load process functions
    process_blocks = [
        get_raw_features,
        get_ce_features,
        get_oe_features,
        get_acquisition_date_feature,
        get_dating_features,
        get_sub_title_features,
        get_basic_text_features,
        get_text_vec_features,
        get_agg_nunique_principal_maker_features,
        get_agg_principal_maker_features,
        get_agg_art_series_id_features,
        # get_agg_dating_sorting_date_features,
        get_cross_num_features,
        # get_te_features,
        get_w2v_tec_features,
        get_w2v_occupation_features,
        get_w2v_col_features,
        get_w2v_material_features,
        get_w2v_production_place_features,
        get_w2v_historical_person_features,
        get_w2v_hist_mat_col_tech_pro_acq_features,
        get_w2v_hist_mat_col_tec_pro_features,
        get_w2v_mat_col_tec_features,
        get_w2v_mat_col_tec_pro_features,
        get_w2v_col_tec_pro_features,
        get_w2v_pro_tec_features,
        get_w2v_hist_tec_features,
        get_w2v_mat_pro_tec_features,
        get_w2v_mat_tec_features,
        get_w2v_col_tec_features,
        get_w2v_hist_col_tec_features,
        get_w2v_tec_col_acq_features,
        get_w2v_dating_year_features,
        # get_w2v_palette_features,

        # get_doc2vec_features
    ]

    # preprocess
    output_df = []
    for func in tqdm(process_blocks):
        _df = func(input_df)
        output_df.append(_df)
    output_df = pd.concat(output_df, axis=1)

    # separate train and test
    train_x = output_df.iloc[:len(train)]
    if config.TARGET in train_x.columns:
        train_x.drop(config.TARGET, axis=1, inplace=True)
    test_x = output_df.iloc[len(train):].reset_index(drop=True)
    if config.TARGET in test_x.columns:
        test_x.drop(config.TARGET, axis=1, inplace=True)
    train_y = train[config.TARGET]
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x}


# ------------------------------------------------------------------------------- #
# ------ Train & predict ---------


# get train
def get_train_data(train, test):
    all_features = preprocess(train, test)
    x = all_features["train_x"]
    y = all_features["train_y"]
    return x, y


# get test
def get_test_data(train, test):
    all_features = preprocess(train, test)
    x = all_features["test_x"]
    return x


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
    test = pd.read_csv(os.path.join(config.INPUT, "test.csv"))

    # preprocess
    train_x, train_y = get_train_data(train, test)
    test_x = get_test_data(train, test)

    # model
    model = MyLGBMModel

    # set run params
    run_params = {
        "metrics": mean_squared_error,
        "cv": make_gkf,
        "feature_select_method": "tree_importance",
        "feature_select_fold": 5,
        "feature_select_num": 500,
        "folds": 5,
        "seeds": [0, 1, 2, 3, 4],
    }

    # set model params
    model_params = {
        "n_estimators": 10000,
        "objective": 'regression',
        "learning_rate": 0.01,
        "num_leaves": 31,
        "random_state": 2021,
        "n_jobs": -1,
        "importance_type": "gain",
        'colsample_bytree': .5,
        "reg_lambda": 5,
    }
    # fit params
    fit_params = {
        "early_stopping_rounds": 200,
        "verbose": 100
    }

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
