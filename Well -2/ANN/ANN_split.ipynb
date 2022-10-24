import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
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

train_data, test_data, train_labels, test_labels = train_test_split(x, y,test_size=0.3)

ann_reg = MLPRegressor(hidden_layer_sizes= (500, 100, 500), activation= ('relu'), solver= 'adam') 
ann_reg.fit(train_data, train_labels)
y_pred_test = ann_reg.predict(test_data)
y_pred_train = ann_reg.predict(train_data)

plt.scatter(test_labels,y_pred_test)
plt.title("ANN Result for SFT Testing data")
plt.ylabel("SFT Real")
plt.xlabel("SFT predicted")
plt.show()

plt.scatter(train_labels,y_pred_train)
plt.title("ANN Result for SFT Training data")
plt.ylabel("SFT Real")
plt.xlabel("SFT predicted")
plt.show()
R2_train = r2_score(train_labels, y_pred_train)
RMS_train = mean_squared_error(train_labels, y_pred_train, squared=False)
R2_test = r2_score(test_labels, y_pred_test)
RMS_test = mean_squared_error(test_labels, y_pred_test, squared=False)
MAE_train= mean_absolute_error(train_labels, y_pred_train)
MAE_test= mean_absolute_error(test_labels, y_pred_test)
MAPE_train= mean_absolute_percentage_error(train_labels, y_pred_train)
MAPE_test= mean_absolute_percentage_error(test_labels, y_pred_test)