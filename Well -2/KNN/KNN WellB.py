
import pandas as pd
from sklearn.utils import shuffle

#Read dataset
dff = pd.read_csv('KNN_b.csv') 
df= pd.DataFrame(dff, columns= ['Time','BHT','Mud Inlet Temp','Mud Outlet Temp','SFT'])
df = shuffle(df)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x= df.drop('SFT', axis=1)
x_scaled = scaler.fit_transform(x) # Apply the transformation
x = pd.DataFrame(x_scaled) # Convert to DataFrame format
y = df['SFT']

#Data Cleaning
# Aa=All.dropna() # option 1
# Aa=All.drop("Avg Wind Speed at 100m (m/s)", axis=1) # option 2
# median = All["Avg Wind Speed at 100m (m/s)"].median() # option 3
# All["Avg Wind Speed at 100m (m/s)"].fillna(median, inplace=True)


#import required packages
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 100)}
knn = neighbors.KNeighborsRegressor()
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
#fit model to data
knn_gscv.fit(x,y)
#check top performing n_neighbors value
print(knn_gscv.best_params_)
