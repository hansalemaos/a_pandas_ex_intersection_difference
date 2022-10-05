### Computes the intersection/symmetric difference of n DataFrames/Series

#### Installation

```python
pip install a-pandas-ex-intersection-difference
```

#### Usage

```python
from  a_pandas_ex_intersection_difference import pd_add_set
pd_add_set()
import pandas as pd
```

##### THE CODE ABOVE WILL ADD SOME METHODS TO! YOU CAN USE PANDAS LIKE YOU DID BEFORE, BUT YOU WILL HAVE A COUPLE OF METHODS MORE:



- pandas.DataFrame.**ds_set_intersections** / pandas.Series.**ds_set_intersections**
- pandas.DataFrame.**ds_set_symmetric_difference** / pandas.Series.**ds_set_symmetric_difference**
- pandas.**DataFrame.ds_set_union** / **pandas.Series.ds_set_union**
- pandas.DataFrame.**ds_value_counts_to_column** / pandas.Series.**ds_value_counts_to_column**



##### pandas.DataFrame.**ds_set_intersections** / pandas.Series.**ds_set_intersections**

```python
    #Computes the intersection of n DataFrames/Series

    #Example
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
```

##### pandas.DataFrame.**ds_set_symmetric_difference** / pandas.Series.**ds_set_symmetric_difference**

```python
    #Computes the symmetric difference of n DataFrames/Series

    #Example
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
```

##### pandas.**DataFrame.ds_set_union** / **pandas.Series.ds_set_union**

```python
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


    #If, for whatever reason, you don't want to use pd.concat(), you can use this method.
    #Don't use this method if you can use pd.concat

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
```

##### pandas.DataFrame.**ds_value_counts_to_column** / pandas.Series.ds_value_counts_to_column

```python
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
```
