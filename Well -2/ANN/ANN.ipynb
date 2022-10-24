
import pandas as pd
from sklearn.utils import shuffle

#Read dataset
dff = pd.read_csv('ANN_B.csv') 
df= pd.DataFrame(dff, columns= ['Time','BHT','Mud Inlet Temp','Mud Outlet Temp','SFT'])
df = shuffle(df)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x= df.drop('SFT', axis=1)
x_scaled = scaler.fit_transform(x) # Apply the transformation
x = pd.DataFrame(x_scaled) # Convert to DataFrame format
y = df['SFT']


from sklearn.model_selection import GridSearchCV


from sklearn.neural_network import MLPRegressor
param_grid = {
    'hidden_layer_sizes': [(50,100,50),(25,50,25),(100,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['adam'],
}
MLP = MLPRegressor(random_state=1, max_iter=1000)
MLP_Regressor = GridSearchCV(MLP, param_grid, cv=5)
MLP_results_ = MLP_Regressor.fit(x,y)
print(MLP_results_.best_params_)

