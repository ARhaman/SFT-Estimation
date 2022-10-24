import pandas as pd
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

#Read dataset
dff = pd.read_csv('RF_A.csv') 
df= pd.DataFrame(dff, columns= ['Time','BHT','Mud Inlet Temp','Mud Outlet Temp','SFT'])
df = shuffle(df)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x= df.drop('SFT', axis=1)
x_scaled = scaler.fit_transform(x) # Apply the transformation
x = pd.DataFrame(x_scaled) # Convert to DataFrame format
y = df['SFT']

#import required packages
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

# Set the parameters by cross-validation
tuned_parameters = {'n_estimators': [500, 700, 1000, 1500], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [1, 2, 3]}

# clf = ensemble.RandomForestRegressor(n_estimators=500, n_jobs=1, verbose=1)
clf = GridSearchCV(ensemble.RandomForestRegressor(), tuned_parameters, cv=5, 
                   n_jobs=-1, verbose=1)

print(clf.best_params_)



# yy_pred = pd.DataFrame(y_pred)
# df['Predicted'] = yy_pred

