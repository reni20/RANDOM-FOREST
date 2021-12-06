import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, recall_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn import linear_model
import warnings
import pickle
import xlrd
train = pd.read_excel('mobile phone.xlsx')
train.head()"
def _get_category_mapping(column)
 train['Merk'] = train['Merk'].astype('category')
merk= _get_category_mapping(train['Merk'])
train['Merk'] = train['Merk'].cat.codes
train['Seri'] = train['Seri'].astype('category')
seri = _get_category_mapping(train['Seri'])
train['Seri'] = train['Seri'].cat.codes
train['Memori'] = train['Memori'].astype('category')
memori = _get_category_mapping(train['Memori'])
train['Memori'] = train['Memori'].cat.codes
train['RAM'] = train['RAM'].astype('category')
ram = _get_category_mapping(train['RAM'])
train['RAM'] = train['RAM'].cat.codes"
X = train.iloc[:, 0:4].values
y= train.iloc[:, 4]
# Menjadi dataset ke dalam Training set and Test set\n",
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)
RandomForestClassifier(random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)"
y_pred = model.predict(X_test)"
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
df = pd.DataFrame ({'Real Values': y_test, 'Predicted Values': y_pred}) 
df
import matplotlib.pyplot as plt
# Build a plot
plt.scatter(y_pred, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')
# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
print('MAE:\\t$%.2f' % mean_absolute_error(y_test, y_pred))
print('MSLE:\\t%.5f' % mean_squared_log_error(y_test, y_pred))
from sklearn.metrics import accuracy_score
print('Correct Prediction (%): ', accuracy_score(y_test, model.predict(X_test), normalize=True)*100.0)
  
