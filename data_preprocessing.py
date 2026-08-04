# TODO: настроить логирование в один файл
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler


class DataProcessing:
    __borders = {"sleep_duration": [0.0, 24.0], "heart_rate": [0.0, 220.0], "bmi": [0.0, 100],
        "calorie_expenditure": [0.0, np.inf], "step_count": [0.0, np.inf], 
        "exercise_duration": [0.0, 1440.0], "water_intake": [0.0, np.inf],
        "diet_type": ("veg", "non-veg", "balanced"), "stress_level": ("low", "high", "medium"),
        "sleep_quality": ('average', 'poor', 'good'), "physical_activity_level": ('sedentary', 'moderate', 'active'),
        "smoking_alcohol": ('yes', 'occasional', 'no'), "gender": ('female', 'other', 'male')}
    __columns_categor = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]
    __translate_dict = {'at-risk': 0, 'unhealthy': 1, 'fit': 2} # правило перевода таргетов в числа
    __columns_to_scal = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count", "exercise_duration", "water_intake"] 
    
    def __init__(self, df:pd.DataFrame, data_type="train") -> None:
        self.data_type = data_type    # train/test
        self.median: pd.Series
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = RobustScaler()
        self.df = df

    @classmethod
    def get_translate_dict(cls):
        return cls.__translate_dict

    def __check_valid_of_data(self):
        '''проверяет, что бы заполненные данные лежали в нормальных интервалах'''
        for col_name in self.df.columns:
            if col_name not in DataProcessing.__borders:
                continue
            else:
                try:
                    if type(DataProcessing.__borders[col_name]) is tuple:
                        mask = ~(self.df[col_name].isin(DataProcessing.__borders[col_name]) | self.df[col_name].isna())
                    elif type(DataProcessing.__borders[col_name]) is list:
                        mask = ~(((self.df[col_name] >= DataProcessing.__borders[col_name][0]) & (self.df[col_name] <= DataProcessing.__borders[col_name][1])) | self.df[col_name].isna())
                    else:
                        raise AttributeError("Нестандартный тип данных из словаря borders")
                except AttributeError as ae:
                    print(f"Словарь borders повреждён: {ae}")
                print(f"Столбец {col_name}: {mask.sum()}")

    def __corrupted_data(self):
        '''удаляет сэмплы с 3-мя и более пропусками'''
        self.df = self.df[self.df.isnull().sum(axis=1) < 3]

    def __onehot_and_median(self):
        '''OneHot + mediana'''
        # Заменяем NaN на медианные значения
        if self.data_type == "train": 
            df_Y = self.df[["id", "health_condition"]].copy()
            self.df.drop(columns=["id", "health_condition"], inplace=True)

        df_numerical = self.df.drop(columns=DataProcessing.__columns_categor)

        if self.data_type == "train":
            self.median = df_numerical.median()
        df_numerical = df_numerical.fillna(self.median)

        df_categorical = self.df[DataProcessing.__columns_categor].fillna("Missing")
        if self.data_type == "train":
            df_categorical_encoded = self.encoder.fit_transform(df_categorical)
        else:
            df_categorical_encoded = self.encoder.transform(df_categorical)

        df_categorical = pd.DataFrame(
            df_categorical_encoded,
            columns=self.encoder.get_feature_names_out(DataProcessing.__columns_categor),
            index=df_categorical.index 
        )

        if self.data_type == "train":
            self.df = pd.concat([df_Y, df_numerical, df_categorical], axis=1)
            self.df['health_condition'] = self.df['health_condition'].map(DataProcessing.__translate_dict)

        else:
             self.df = pd.concat([df_numerical, df_categorical], axis=1)

        self.df = self.df.drop(columns=["id"], errors="ignore")

    def __scaler(self):
        if self.data_type == "train":
            scaled_values = self.scaler.fit_transform(self.df[DataProcessing.__columns_to_scal])
            self.df[DataProcessing.__columns_to_scal] = scaled_values
        else:
            scaled_values = self.scaler.transform(self.df[DataProcessing.__columns_to_scal])
            self.df[DataProcessing.__columns_to_scal] = scaled_values

    def df_prepr(self):
        if self.data_type == "train":
            self.__check_valid_of_data()
            self.__corrupted_data()
        self.__onehot_and_median()
        self.__scaler()
        return self.df.copy()