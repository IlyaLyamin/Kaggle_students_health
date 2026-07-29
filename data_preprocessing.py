# TODO: настроить логирование в один файл
import pandas as pd
from sklearn.preprocessing import OneHotEncoder



def df_prepr_for_test(df: pd.DataFrame, 
                      encoder: OneHotEncoder,
                      median: pd.Series) -> pd.DataFrame:
    
    if not isinstance(encoder, OneHotEncoder):
        raise TypeError("encoder must be OneHotEncoder instance")
   
    # =================================================================================================================      
    # используем OneHot и заполнение медианным значением
    columns_categor = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]

    df_numerical = df.drop(columns=columns_categor)
    df_numerical = df_numerical.fillna(median)

    df_categorical = df[columns_categor].fillna("Missing")
    df_categorical_encoded = encoder.transform(df_categorical)

    df_categorical = pd.DataFrame(
        data=df_categorical_encoded,
        columns=encoder.get_feature_names_out(columns_categor),
        index=df_categorical.index
    )

    df = pd.concat([df_numerical, df_categorical], axis=1)

    # =================================================================================================================

    df = df.drop(columns=["id"], errors="ignore")
    
    return df