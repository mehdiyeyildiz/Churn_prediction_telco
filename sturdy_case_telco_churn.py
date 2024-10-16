import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



df = pd.read_csv(r"C:\Users\Mehdiye\Desktop\study case\6 Machine learning\TelcoChurn\Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()
df.describe()


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
label_encoder(df, "Churn")
df["tenure"] = df[df["tenure"] > 0]
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if "Churn" not in col]

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
    num_summary(df, col, plot=True)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "TARGET_COUNT": dataframe.groupby(categorical_col)[target].count()}), end="\n\n\n")
for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

low_limit, up_limit = outlier_thresholds(df, num_cols)
check_outlier(df, num_cols)
missing_values_table(df)

#df.duplicated().sum()

df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "Churn", cat_cols)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols)

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])

df.head()

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

#1.MODEL
rf_model = RandomForestClassifier()
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

rf_params = {"max_depth": [15, 25, None],
             "max_features": [3, 6],
             "min_samples_split": [8, 18, 20, 23],
             "n_estimators": [100, 150, 200]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)



rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.8007946486137975
cv_results['test_f1'].mean()
#0.5601501813806453
cv_results['test_roc_auc'].mean()
#0.8438899267151407


#2.MODEL
gbm_model = GradientBoostingClassifier()

gbm_model.get_params()

cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.8056235482934383
cv_results['test_f1'].mean()
#0.5918585330116308
cv_results['test_roc_auc'].mean()
#0.8450928613375653

gbm_params = {"learning_rate": [0.08, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [100, 150],
              "subsample": [1, 0.5]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.8070437004000258
cv_results['test_f1'].mean()
#0.5963691702619169
cv_results['test_roc_auc'].mean()
#0.8455903194437759

#3.MODEL
xgboost_model = XGBClassifier(use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.7817705981031035
cv_results['test_f1'].mean()
#0.5510214792445778
cv_results['test_roc_auc'].mean()
#0.8220939418767912

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12],
                  "n_estimators": [100, 500],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.8020732184334474
cv_results['test_f1'].mean()
#0.5802096839836393
cv_results['test_roc_auc'].mean()
#0.8462693778634559


#4.MODEL
lgbm_model = LGBMClassifier()
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.7922780300341957
cv_results['test_f1'].mean()
#0.568224959434718
cv_results['test_roc_auc'].mean()
#0.8346033271466494

lgbm_params = {"learning_rate": [0.01, 0.03],
               "n_estimators": [400, 500, 550],
               "colsample_bytree": [0.4, 0.5, 0.7]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.8453337532456466
cv_results['test_f1'].mean()
#0.5861395853659375
cv_results['test_roc_auc'].mean()
#0.804772082069811




def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)



#label encoder sınıfı manuel sıralama
df = pd.read_csv(r"C:\Users\Mehdiye\Desktop\study case\6 Machine learning\TelcoChurn\Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()
df.describe()

df["Churn"] = LabelEncoder().fit_transform(df["Churn"])

siniflar = df["Churn"]
sinif_sirasi = ["Yes","No"] # numaralnadırmasını istedğim sırada girdim.

siniflar_encoded = {sinif_sirasi[i]: i for i in range(len(sinif_sirasi))}
siniflar_encoded_list = [siniflar_encoded[sinif] for sinif in siniflar]

print(siniflar_encoded_list)

#örnek
siniflar = ['a','b', 'c', 'd']

sinif_sirasi = ['b', 'c', 'a', 'd']

siniflar_encoded = {sinif_sirasi[i]: i for i in range(len(sinif_sirasi))}
siniflar_encoded_list = [siniflar_encoded[sinif] for sinif in siniflar]

print(siniflar_encoded_list)