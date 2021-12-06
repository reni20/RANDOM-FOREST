import numpy as np\n",
import pandas as pd\n",
from IPython.display import display\n",
from sklearn.preprocessing import StandardScaler\n",
from sklearn.model_selection import train_test_split\n",
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold\n",
from sklearn.pipeline import Pipeline\n",
from sklearn.metrics import r2_score\n",
from sklearn.ensemble import RandomForestRegressor\n",
import matplotlib.pyplot as plt\n",
from scipy.stats import pearsonr\n",
import seaborn\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn import metrics\n",
    "from sklearn import tree\n",
    "import warnings\n",
    "import pickle\n",
    "import xlrd"
    "df= pd.read_excel('ponsel.xlsx')\n",
    "df.head()"
    "df.describe()"

    "def _get_category_mapping(column):\n",
    "    \"\"\" Return the mapping of a category \"\"\"\n",
    "    return dict([(cat, code) for code, cat in enumerate(column.cat.categories)])\n",
    "df['Merk'] = df['Merk'].astype('category')\n",
    "merk= _get_category_mapping(df['Merk'])\n",
    "df['Merk'] = df['Merk'].cat.codes\n",
    "df['Series'] = df['Series'].astype('category')\n",
    "series = _get_category_mapping(df['Series'])\n",
    "df['Series'] = df['Series'].cat.codes\n",
    "df['Memori'] = df['Memori'].astype('category')\n",
    "memori = _get_category_mapping(df['Memori'])\n",
    "df['Memori'] = df['Memori'].cat.codes\n",
    "df['RAM'] = df['RAM'].astype('category')\n",
    "ram = _get_category_mapping(df['RAM'])\n",
    "df['RAM'] = df['RAM'].cat.codes"
 
    "#Save Variabel kategorik\n",
    "import joblib\n",
    "def _save_variable(variable, filename):\n",
    "    \"\"\" Save a variable to a file \"\"\"\n",
    "    joblib.dump(variable, filename)\n",
    "_save_variable(merk, 'Merk.pkl')\n",
    "_save_variable(series, 'Series.pkl')\n",
    "_save_variable(ram, 'RAM.pkl')\n",
    "_save_variable(memori, 'Memori.pkl')"
    "X = df.iloc[:, 0:4].values\n",
    "y= df.iloc[:, 4]"

    "# Menjadi dataset ke dalam Training set and Test set\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)"
  
    "tree_feature =  pd.Series(model.feature_importances_, FEATURES).sort_values(ascending = True)\n",
    "plt.figure(figsize = (8,8))\n",
    "plt.barh(FEATURES, tree_feature)\n",
    "plt.xlabel('Mean Impurity Reduction', fontsize = 12)\n",
    "plt.ylabel('Features', fontsize = 12)\n",
    "plt.yticks(fontsize = 12)\n",
    "plt.title('Feature Importances', fontsize = 20)"

       "RandomForestRegressor(n_estimators=300, random_state=42)"
    "rf = RandomForestRegressor(n_estimators=300, random_state = 42)\n",
    "rf.fit(X_train, y_train)"
    "_save_variable(model, 'RFR.mdl')
    "FEATURES = [\n",
    "    'Merk',\n",
    "    'Seri',\n",
    "    'RAM',\n",
    "    'Memori',\n",
    "TARGET = 'Harga'\n",
#Variabel importance
features_importance = rf.feature_importances_
print(\"Feature ranking:\")
for i, data_class in enumerate(FEATURES):
    print(\"{}. {} ({})\".format(i + 1, data_class, features_importance[i]))
   
prediksi = rf.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediksi))
r2_score(y_test, prediksi)
# Score model
rf.score(X_train, y_train)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediksi))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediksi))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediksi)))
print('MAE:\\t$%.2f' % mean_absolute_error(y_test, prediksi))
print('MSLE:\\t%.5f' % mean_squared_log_error(y_test, prediksi))
  
df1 = pd.DataFrame ({'Real Values': y_test, 'Predicted Values': prediksi})
df1
 df1.describe()
 def simple_scatter_plot(y_test, prediksi, output_filename, title_name, x_axis_label, y_axis_label)
   \"\"\"Simple scatter plot.
      Args:
           x_data (list): List with x-axis data.
            y_data (list): List with y-axis data.
           output_filename (str): Path to output image in PNG format.
            title_name (int): Plot title.\n",
            x_axis_label (str): X-axis Label.
            y_axis_label (str): Y-axis Label.

 seaborn.set(color_codes=True)
 matplotlib.figure(1, figsize=(9, 6))
matplotlib.title(title_name)\n",
 ax = seaborn.scatterplot(x=y_test, y=prediksi)
ax.set(xlabel=x_axis_label, ylabel=y_axis_label)
 matplotlib.savefig(output_filename, bbox_inches='tight', dpi=300)
 matplotlib.close()
  
 # find the correlation between real answer and prediction
  correlation = round(pearsonr(prediksi, y_test)[0], 5)
 output_filename = \"rf_regression.png\
 title_name = \"Random Forest Regression - Real Ponsel Price vs Predicted Ponsel Price - correlation ({})\".format(correlation)
  x_axis_label = \"Real Ponsel Price"
  y_axis_label = \"Predicted Ponsel Price"
# plot data\n",
   simple_scatter_plot(y_test, prediksi, output_filename, title_name, x_axis_label, y_axis_label)
  
# Build a plot\n",
plt.scatter(prediksi, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')
# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()

# Pull out one tree from the forest
  tree = rf.estimators_[4]
  
 from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names =None, 
   rounded = True, precision = 1)
 
import os
 os.environ[\"PATH\"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
 graph.write_png('tree.png')
RandomForestRegressor(max_depth=4, n_estimators=300)
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=300, max_depth = 4)
rf_small.fit(X_train, y_train)"
  
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = None, rounded = True, precision = 1)
 
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
 graph.write_png('small_tree3.png')
# make a single prediction
   row = [[4,8,2,5]]
   yhat = rf_small.predict(row)
    print('Prediction: %d' % yhat[0])
   
