import pandas as pd
import numpy as np
import category_encoders as ce
import joblib as jb
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('male_players (legacy).csv', low_memory=False)

# The non-relevant columns to drop
# These columns would not be relevant to predict a player's overall rating


irrelevant_cols = ['player_id','player_url','fifa_version', 'fifa_update','fifa_update_date','dob', 'short_name',
                   'long_name', 'league_id','league_name', 'club_team_id', 'club_name', 'club_jersey_number',
                   'club_loaned_from', 'club_joined_date', 'club_contract_valid_until_year', 'nationality_id',
                   'nation_jersey_number', 'body_type', 'real_face', 'player_tags', 'player_traits', 'player_face_url']

dataset.drop(columns = irrelevant_cols, axis=True, inplace=True)

#Dropping all columns that have more than 30% null values

L = []
L_less = []
for i in dataset.columns:
    if((dataset[i].isnull().sum()) < (0.3*(dataset.shape[0]))):
        L.append(i)
    else:
        L_less.append(i)

dataset = dataset[L]

dataset =dataset.iloc[:, :56]
print(dataset.info())

#Separate the quantitative and categorical variables
numeric_data = dataset.select_dtypes(include = np.number)
non_numeric = dataset.select_dtypes(include = ['object'])


# Multivariate imputation

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter = 10, random_state = 0)
numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)), columns = numeric_data.columns)

print(numeric_data)

corr_matrix = numeric_data.corr()

#correlation between all the columns listed and overall rating
corr_matrix["overall"].sort_values(ascending=False)


from sklearn.impute import SimpleImputer
# median for missing numeric values
# mean for missing object values
num_si = SimpleImputer(strategy='median')
obj_si = SimpleImputer(strategy='most_frequent')

numeric_scaled = num_si.fit_transform(numeric_data)
non_numeric_scaled = obj_si.fit_transform(non_numeric)

encoder = ce.BinaryEncoder(cols=non_numeric.columns)
col_encoded = encoder.fit_transform(non_numeric)
non_numeric.dropna(inplace=True)

dataset = pd.concat([numeric_data, non_numeric], axis=1).reset_index(drop=True)

dataset = pd.concat([numeric_data, non_numeric], axis=1).reset_index(drop=True)

y = dataset.overall
X = dataset.drop('overall', axis=1)

# scaling

from sklearn.preprocessing import StandardScaler


# Step 1: Select only numeric columns
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
X_numeric = X[numeric_columns]
X_numeric = X_numeric.fillna(X_numeric.mean())
scale = StandardScaler()
scaled = scale.fit_transform(X_numeric)


X = pd.DataFrame(scaled, columns=numeric_columns)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42)

# Multilinear Regression

from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(Xtrain,Ytrain)
y_pred = l.predict(Xtest)

intercept = l.intercept_


coefficients = l.coef_

print(f"Intercept: {intercept}")
print(f"Coefficients: {coefficients}")

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")


'''Polynomial Regression With Degree of 1'''
from sklearn.preprocessing import PolynomialFeatures


poly = PolynomialFeatures(degree=1)
X_poly_train = poly.fit_transform(Xtrain)
X_poly_test = poly.fit_transform(Xtest)


model = LinearRegression()
model.fit(X_poly_train,Ytrain)


y_pred = model.predict(X_poly_test)

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")

'''Polynomial Regression With Degree of 2'''
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(Xtrain)
X_poly_test = poly.fit_transform(Xtest)


model = LinearRegression()
model.fit(X_poly_train,Ytrain)


y_pred = model.predict(X_poly_test)

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")


'''Regularisation Models With Ridge Model'''
from sklearn.linear_model import ElasticNet, Lasso, Ridge

# Train the model
modl = Ridge()
modl.fit(Xtrain, Ytrain)

# Test the model
y_pred = modl.predict(Xtest)

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")


'''Regularisation Models With Lasso Model'''
model = Lasso()
model.fit(Xtrain, Ytrain)

# Tsst the model
y_pred = model.predict(Xtest)

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")


'''Regularisation Models With ElasticNet Model'''
model = ElasticNet()
model.fit(Xtrain, Ytrain)

# Test the model
y_pred = model.predict(Xtest)

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")


'''Decision Tree Regression'''
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(max_depth=12)
dtree.fit(Xtrain, Ytrain)

y_pred = dtree.predict(Xtest)
print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")

# Ensemble Learning
# imported several packages for different ensemble models to be created

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
import pickle as pkl
from sklearn.ensemble import VotingClassifier, VotingRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

dt=DecisionTreeRegressor(criterion='absolute_error')
knn=KNeighborsRegressor(n_neighbors=7)

voting=VotingRegressor(estimators=[('knn', knn), ('l', l), ('Dtree', dtree)])
voting.fit(Xtrain, Ytrain)

y_pred = voting.predict(Xtest)

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")

import xgboost as xgb

xgb_reg = xgb.XGBRegressor()

xgb_reg.fit(Xtrain, Ytrain)

y_pred = xgb_reg.predict(Xtest)

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")

voting=VotingRegressor(estimators=[('xbg', xgb_reg), ('Dtree', dtree)])
voting.fit(Xtrain, Ytrain)

y_pred = voting.predict(Xtest)

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")

jb.dump(voting, 'best_model.pkl', compress = 9)


# testing the model
testor = pd.read_csv('players_22.csv', low_memory='False')

# Dropping all columns that have more than 30% null values

test_L = []
test_L_less = []
for i in testor.columns:
    if((testor[i].isnull().sum()) < (0.3*(testor.shape[0]))):
        test_L.append(i)
    else:
        test_L_less.append(i)


testor = testor[test_L]
testor = testor[dataset.columns]

#Separate the quantitative and categorical variables
test_numeric = testor.select_dtypes(include = np.number)
test_non_numeric = testor.select_dtypes(include = ['object'])

imp = IterativeImputer(max_iter = 10, random_state = 0)
test_numeric = pd.DataFrame(np.round(imp.fit_transform(test_numeric)), columns = test_numeric.columns)

encoder = ce.BinaryEncoder(cols=test_non_numeric.columns)
col_encoded = encoder.fit_transform(test_non_numeric)
test_non_numeric.dropna(inplace=True)


dataset = pd.concat([test_numeric, test_non_numeric], axis=1).reset_index(drop=True)

# Testing

X1 = testor.drop(['overall'], axis=1)
y1 = testor['overall']

# Step 1: Select only numeric columns
numeric_columns = X1.select_dtypes(include=['float64', 'int64']).columns
X1_numeric = X1[numeric_columns]
X1_numeric = X1_numeric.fillna(X1_numeric.mean())
scale = StandardScaler()
scaled = scale.fit_transform(X1_numeric)


X1 = pd.DataFrame(scaled, columns=numeric_columns)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X1,y1,test_size=0.2,random_state=42)

Xtrain = scale.fit_transform(Xtrain)
Xtest = scale.fit_transform(Xtest)

y_pred = voting.predict(Xtest)

print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")

jb.dump(scale, 'scaler.pkl')

