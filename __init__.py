from functools import reduce
from itertools import permutations
from typing import Union

import pandas as pd
from a_pandas_ex_plode_tool import all_nans_in_df_to_pdNA
from a_pandas_ex_df_to_string import ds_to_string
from pandas.core.base import PandasObject


def qq_s_value_counts_to_column(df: pd.Series) -> pd.Series:
    """
    df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")

    df2.Sex.ds_value_counts_to_column()
         PassengerId  Survived  Pclass  ...      Fare Cabin  Embarked
    504          505         1       1  ...   86.5000   B79         S
    781          782         1       1  ...   57.0000   B20         S
    855          856         1       3  ...    9.3500   NaN         S
    552          553         0       3  ...    7.8292   NaN         Q
    777          778         1       3  ...   12.4750   NaN         S
    ..           ...       ...     ...  ...       ...   ...       ...
    756          757         0       3  ...    7.7958   NaN         S
    224          225         1       1  ...   90.0000   C93         S
    488          489         0       3  ...    8.0500   NaN         S
    309          310         1       1  ...   56.9292   E36         C
    581          582         1       1  ...  110.8833   C68         C
    [446 rows x 12 columns]

    df2.Sex.ds_value_counts_to_column()
    Out[22]:
    0      152
    1      152
    2      152
    3      294
    4      152
          ...
    441    294
    442    294
    443    294
    444    152
    445    152
    Name: 0, Length: 446, dtype: int64

    This method could also be useful, when you are comparing DataFrames, since it counts the different values in a Series
    and returns a DataFrame that you can merge with your original DataFrame
        Parameters
            df: pd.Series
        Returns
            pd.DataFrame
    """
    series_ = df.copy()
    try:
        return (
            pd.Series(series_.value_counts().to_dict())
            .reindex(series_)
            .to_frame()
            .reset_index()[0]
        )
    except Exception:

        series_ = series_.qq_ds_to_string()
        return (
            pd.Series(series_.value_counts().to_dict())
            .reindex(series_)
            .to_frame()
            .reset_index()[0]
        )


def filter_same_dfs_columns(*args) -> list:
    args_ = [x.to_frame().copy() if isinstance(x, pd.Series) else x for x in args]
    comuncols = list(
        reduce(set.intersection, [set(x.columns.to_list()) for x in args_])
    )
    passdfs = [dfaf[[x for x in comuncols]].copy() for dfaf in args_]
    return passdfs


def set_intersections_df(
    *args, accept_df_with_different_columns: bool = True
) -> pd.DataFrame:
    """
    Computes the intersection of n DataFrames/Series

    Example
    df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")

    #Let's create some DataFrames with random data from df

    df1 = df.sample(len(df) - len(df)//2).copy()
    df2 = df.sample(len(df) - len(df)//2).copy()
    df3 = df.sample(len(df) - len(df)//2).copy()
    df4 = df.sample(len(df) - len(df)//2).copy()
    df5 = df.sample(len(df) - len(df)//2).copy()

    df1.ds_set_intersections(df2) #Comparing 2 DataFrames
    Out[14]:
         Parch  PassengerId      Fare  Survived  ...  SibSp Embarked     Sex Cabin
    0        1          802   26.2500         1  ...      1        S  female   NaN
    1        0          506  108.9000         0  ...      1        C    male   C65
    2        0          386   73.5000         0  ...      0        S    male   NaN
    3        0          621   14.4542         0  ...      1        C    male   NaN
    4        1          273   19.5000         1  ...      0        S  female   NaN
    ..     ...          ...       ...       ...  ...    ...      ...     ...   ...
    439      0          240   12.2750         0  ...      0        S    male   NaN
    440      0          235   10.5000         0  ...      0        S    male   NaN
    441      1          269  153.4625         1  ...      0        S  female  C125
    442      0          394  113.2750         1  ...      1        C  female   D36
    443      0          400   12.6500         1  ...      0        S  female   NaN
    [444 rows x 12 columns]
    df1.ds_set_intersections(df2,df3)  #Comparing 3 DataFrames
    Out[15]:
         Parch  PassengerId      Fare  Survived  ...  SibSp Embarked     Sex Cabin
    0        0          506  108.9000         0  ...      1        C    male   C65
    1        1          480   12.2875         1  ...      0        S  female   NaN
    2        1          581   30.0000         1  ...      1        S  female   NaN
    3        1          447   19.5000         1  ...      0        S  female   NaN
    4        0           16   16.0000         1  ...      0        S  female   NaN
    ..     ...          ...       ...       ...  ...    ...      ...     ...   ...
    340      2          154   14.5000         0  ...      0        S    male   NaN
    341      0          668    7.7750         0  ...      0        S    male   NaN
    342      0          702   26.2875         1  ...      0        S    male   E24
    343      0          610  153.4625         1  ...      0        S  female  C125
    344      0          450   30.5000         1  ...      0        S    male  C104
    [345 rows x 12 columns]
    df1.ds_set_intersections(df2,df3, df4)  #Comparing 4 DataFrames
    Out[16]:
         Parch  PassengerId      Fare  Survived  ...  SibSp Embarked     Sex Cabin
    0        0          506  108.9000         0  ...      1        C    male   C65
    1        1          581   30.0000         1  ...      1        S  female   NaN
    2        0          283    9.5000         0  ...      0        S    male   NaN
    3        0          488   29.7000         0  ...      0        C    male   B37
    4        0          610  153.4625         1  ...      0        S  female  C125
    ..     ...          ...       ...       ...  ...    ...      ...     ...   ...
    227      0           23    8.0292         1  ...      0        Q  female   NaN
    228      1          619   39.0000         1  ...      2        S  female    F4
    229      2          473   27.7500         1  ...      1        S  female   NaN
    230      0          253   26.5500         0  ...      0        S    male   C87
    231      0          618   16.1000         0  ...      1        S  female   NaN
    [232 rows x 12 columns]
    df1.ds_set_intersections(df2,df3, df4, df5)  #Comparing 5 DataFrames
    Out[17]:
         Parch  PassengerId      Fare  Survived  ...  SibSp Embarked     Sex Cabin
    0        0          506  108.9000         0  ...      1        C    male   C65
    1        1          581   30.0000         1  ...      1        S  female   NaN
    2        1           17   29.1250         0  ...      4        Q    male   NaN
    3        2           59   27.7500         1  ...      1        S  female   NaN
    4        0          463   38.5000         0  ...      0        S    male   E63
    ..     ...          ...       ...       ...  ...    ...      ...     ...   ...
    140      2          166   20.5250         1  ...      0        S    male   NaN
    141      0          705    7.8542         0  ...      1        S    male   NaN
    142      1           51   39.6875         0  ...      4        S    male   NaN
    143      0          833    7.2292         0  ...      0        C    male   NaN
    144      2          154   14.5000         0  ...      0        S    male   NaN
    [145 rows x 12 columns]


    """
    if accept_df_with_different_columns:
        args = filter_same_dfs_columns(*args)
    originalcolumns = args[0].columns
    dfa = set_check_dfs(*args, setfunction=set.intersection)
    return dfa.filter(originalcolumns)


def set_symmetric_difference_df(
    *args, accept_df_with_different_columns: bool = True
) -> pd.DataFrame:
    """
    Computes the symmetric difference of n DataFrames/Series

    Example
    df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")

    #Let's create some DataFrames with random data from df

    df1 = df.sample(len(df) - len(df)//2).copy()
    df2 = df.sample(len(df) - len(df)//2).copy()
    df3 = df.sample(len(df) - len(df)//2).copy()
    df4 = df.sample(len(df) - len(df)//2).copy()
    df5 = df.sample(len(df) - len(df)//2).copy()

    df1.ds_set_symmetric_difference(df2) #Comparing 2 DataFrames
    Out[18]:
         Parch  PassengerId      Fare  ...  Embarked     Sex        Cabin
    0        0          567    7.8958  ...         S    male          NaN
    1        0           46    8.0500  ...         S    male          NaN
    2        2          342  263.0000  ...         S  female  C23 C25 C27
    3        0          845    8.6625  ...         S    male          NaN
    4        0            1    7.2500  ...         S    male          NaN
    ..     ...          ...       ...  ...       ...     ...          ...
    219      0          865   13.0000  ...         S    male          NaN
    220      5          639   39.6875  ...         S  female          NaN
    221      0           30    7.8958  ...         S    male          NaN
    222      0          332   28.5000  ...         S    male         C124
    223      0          884   10.5000  ...         S    male          NaN
    [448 rows x 12 columns]
    df1.ds_set_symmetric_difference(df2,df3)  #Comparing 3 DataFrames
    Out[19]:
         Parch  PassengerId     Fare  Survived  ...  SibSp Embarked     Sex Cabin
    0        0          567   7.8958         0  ...      0        S    male   NaN
    1        0           46   8.0500         0  ...      0        S    male   NaN
    2        0          845   8.6625         0  ...      0        S    male   NaN
    3        0          142   7.7500         1  ...      0        S  female   NaN
    4        0          579  14.4583         0  ...      1        C  female   NaN
    ..     ...          ...      ...       ...  ...    ...      ...     ...   ...
    106      0          430   8.0500         1  ...      0        S    male   E10
    107      1          363  14.4542         0  ...      0        C  female   NaN
    108      1          531  26.0000         1  ...      1        S  female   NaN
    109      0          748  13.0000         1  ...      0        S  female   NaN
    110      0          876   7.2250         1  ...      0        C  female   NaN
    [339 rows x 12 columns]
    df1.ds_set_symmetric_difference(df2,df3,df4)  #Comparing 4 DataFrames
    Out[20]:
        Parch  PassengerId      Fare  Survived  ...  SibSp Embarked     Sex Cabin
    0       0          567    7.8958         0  ...      0        S    male   NaN
    1       0           46    8.0500         0  ...      0        S    male   NaN
    2       0          142    7.7500         1  ...      0        S  female   NaN
    3       0          579   14.4583         0  ...      1        C  female   NaN
    4       0          365   15.5000         0  ...      1        Q    male   NaN
    ..    ...          ...       ...       ...  ...    ...      ...     ...   ...
    39      2          551  110.8833         1  ...      0        C    male   C70
    40      0           19   18.0000         0  ...      1        S  female   NaN
    41      0          615    8.0500         0  ...      0        S    male   NaN
    42      0          204    7.2250         0  ...      0        C    male   NaN
    43      1          375   21.0750         0  ...      3        S  female   NaN
    [204 rows x 12 columns]
    df1.ds_set_symmetric_difference(df2,df3,df4,df5)  #Comparing 5 DataFrames
    Out[21]:
        Parch  PassengerId     Fare  Survived  ...  SibSp Embarked     Sex Cabin
    0       0          567   7.8958         0  ...      0        S    male   NaN
    1       0          579  14.4583         0  ...      1        C  female   NaN
    2       0          365  15.5000         0  ...      1        Q    male   NaN
    3       0          644  56.4958         1  ...      0        S    male   NaN
    4       0          708  26.2875         1  ...      0        S    male   E24
    ..    ...          ...      ...       ...  ...    ...      ...     ...   ...
    25      0          343  13.0000         0  ...      0        S    male   NaN
    26      0          656  73.5000         0  ...      2        S    male   NaN
    27      0          407   7.7500         0  ...      0        S    male   NaN
    28      0          301   7.7500         1  ...      0        Q  female   NaN
    29      0          819   6.4500         0  ...      0        S    male   NaN
    [125 rows x 12 columns]

        Parameters
            args: Union[pd.Series, pd.DataFrame]
                DataFrames or Series, how many you want
            accept_df_with_different_columns: bool=True
                Let's say you have one DataFrame whose columns are:  [Parch,  PassengerId, Fare, Survived, SibSp,Embarked,  Sex, Cabin]
                If you want to compare it to: [Flight, Fare, Survived, SibSp,Embarked,  Sex, Cabin]
                It won't work, unless you pass accept_df_with_different_columns=True
                Only the columns that are in all dataframes will be compared

        Returns
            pd.DataFrame


    """
    if accept_df_with_different_columns:
        args = filter_same_dfs_columns(*args)
    originalcolumns = args[0].columns
    updateddfs = []
    for_settemp = "for_set____________________"
    test = list([ds_to_string(all_nans_in_df_to_pdNA(x)) for x in args])
    for df1s in test:
        df1s[for_settemp] = df1s.apply(
            lambda x: str(x.__array__()[1:].tolist()), axis=1
        )
        updateddfs.append(df1s.copy())
    perm = permutations([x[for_settemp].to_list() for x in updateddfs])
    onlyones = []
    for i in list(perm):
        half_results = list(reduce(set.symmetric_difference, [set(x) for x in i]))
        if any(half_results):
            for single_result in half_results:
                together = [
                    True if sublist.count(single_result) > 0 else False for sublist in i
                ]
                isgoodresult = together.count(True)
                if isgoodresult == 1:
                    onlyones.append(single_result)
    onlyones = list(set(onlyones))
    allgoodindex = []
    for ini, dataf in enumerate(updateddfs):
        goodindex = dataf.loc[dataf[for_settemp].isin(onlyones)].index
        tmpdf = args[ini].loc[goodindex].copy()
        tmpdf["aa_original_index"] = tmpdf.index.__array__().copy()
        tmpdf["aa_dfposition"] = ini
        allgoodindex.append(tmpdf.reset_index(drop=True).copy())

    return pd.concat(allgoodindex).filter(originalcolumns)


def set_union_df(*args, accept_df_with_different_columns: bool = True) -> pd.DataFrame:
    """

    df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")

    #Let's create some DataFrames with random data from df

    df1 = df.sample(len(df) - len(df)//2).copy()
    df2 = df.sample(len(df) - len(df)//2).copy()
    df3 = df.sample(len(df) - len(df)//2).copy()
    df4 = df.sample(len(df) - len(df)//2).copy()
    df5 = df.sample(len(df) - len(df)//2).copy()


    df1[['PassengerId','Survived','Name']].ds_set_union(df2[['Pclass','Cabin','Name']])
    Out[17]:
                                                      Name
    0                                Carbines, Mr. William
    1                            Sundman, Mr. Johan Julian
    2                                     Dimic, Mr. Jovan
    3                          Harder, Mr. George Achilles
    4                                 Rice, Master. Eugene
    ..                                                 ...
    887                       Carlsson, Mr. August Sigfrid
    888                       Hoyt, Mr. Frederick Maxfield
    889                      Somerton, Mr. Francis William
    890                     Francatelli, Miss. Laura Mabel
    891  Thayer, Mrs. John Borland (Marian Longstreth M...


    If, for whatever reason, you don't want to use pd.concat(), you can use this method.
    Don't use this method if you can use pd.concat

        Parameters
            args: Union[pd.Series, pd.DataFrame]
                DataFrames or Series, how many you want
            accept_df_with_different_columns: bool=True
                Let's say you have one DataFrame whose columns are:  [Parch,  PassengerId, Fare, Survived, SibSp,Embarked,  Sex, Cabin]
                If you want to compare it to: [Flight, Fare, Survived, SibSp,Embarked,  Sex, Cabin]
                It won't work, unless you pass accept_df_with_different_columns=True
                Only the columns that are in all dataframes will be compared

        Returns
            pd.DataFrame


    """
    if accept_df_with_different_columns:
        args = filter_same_dfs_columns(*args)
    originalcolumns = args[0].columns

    dfa = set_check_dfs(*args, setfunction=set.union)
    return dfa.filter(originalcolumns)


def set_check_dfs(*args, setfunction) -> pd.DataFrame:
    for_settemp = "for_set____________________"

    togethercols = args[0].columns.to_list()
    alldataframesconverted = []
    for dfr in args:
        dfr1 = all_nans_in_df_to_pdNA(dfr)
        df1s = ds_to_string(dfr1)
        alldataframesconverted.append(df1s.copy())
    alldataframesconverted3tmp = []
    for col in togethercols:
        allcols = [x[col] for x in alldataframesconverted]
        results1 = list(reduce(setfunction, [set(x.tolist()) for x in allcols]))
        for datafr in alldataframesconverted:
            alldataframesconverted3tmp.append(
                datafr.loc[datafr[col].isin(results1)].copy()
            )
        alldataframesconverted = alldataframesconverted3tmp.copy()
        alldataframesconverted3tmp.clear()

    updateddfs = []
    for df1s in alldataframesconverted:
        df1s[for_settemp] = df1s.apply(
            lambda x: str(x.__array__()[1:].tolist()), axis=1
        )
        updateddfs.append(df1s.copy())
    results1 = list(
        reduce(setfunction, [set(x[for_settemp].to_list()) for x in updateddfs])
    )
    dict_final = {}
    for ini, df1s in enumerate(updateddfs):
        dict_final[ini] = df1s.loc[df1s[for_settemp].isin(results1)].index
    dict_final_dfs = []
    for key, item in dict_final.items():
        tempdf = args[key].loc[item].copy()
        tempdf["aa_original_index"] = tempdf.index.__array__().copy()
        tempdf["aa_dfposition"] = key

        dict_final_dfs.append(tempdf.copy())

    return pd.concat(dict_final_dfs, ignore_index=True)


def series_to_dataframe(
    df: Union[pd.Series, pd.DataFrame]
) -> (Union[pd.Series, pd.DataFrame], bool):
    dataf = df.copy()
    isseries = False
    if isinstance(dataf, pd.Series):
        columnname = dataf.name
        dataf = dataf.to_frame()

        try:
            dataf.columns = [columnname]
        except Exception:
            dataf.index = [columnname]
            dataf = dataf.T
        isseries = True

    return dataf, isseries


def pd_add_set():
    PandasObject.ds_value_counts_to_column = qq_s_value_counts_to_column
    PandasObject.ds_set_intersections = set_intersections_df
    PandasObject.ds_set_symmetric_difference = set_symmetric_difference_df
    PandasObject.ds_set_union = set_union_df
