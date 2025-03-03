# for manupilating data
import pandas as pd
import numpy as np

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
sns.set_style("darkgrid")
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import cross_val_predict
# ml tools
from sklearn.model_selection import train_test_split # to split the data as obvious
from sklearn.preprocessing import StandardScaler # to scale our features

# models
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import (AdaBoostRegressor,
                              RandomForestRegressor,
                              ExtraTreesRegressor, 
                              GradientBoostingRegressor)
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  Lasso,
                                  ElasticNet)

# metrics
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_predict

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
sns.set_style("darkgrid")

# Încărcarea setului de date
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car name"]
df = pd.read_csv(url, delim_whitespace=True, names=columns, na_values='?')
st.title("Vizualizarea datelor")
st.dataframe(df.head()) 

##changing ? to nan in all the columns
df.replace('?',np.nan, inplace= True)

## finding null values in all columns
st.title("Valori nule")
st.write(df.isnull().sum())

st.title("Tip de date")
st.write(df.dtypes)

df["horsepower"] = pd.to_numeric(df["horsepower"], errors = "coerce")
df["origin"] = df["origin"].astype("object")

# test
assert df["horsepower"].dtype == "float64"
assert df["origin"].dtype == "object"

filt = df["horsepower"].isna()
df[filt]

horse_pwr_median = df["horsepower"].median()
df["horsepower"].fillna(horse_pwr_median, inplace = True)

# test
assert df["horsepower"].isna().sum() == 0

# I will see how many duplicates are in the data.
df.duplicated().sum()

# I will search for duplicates in this subset, since it will be the same car.
df.duplicated(subset = ["car name", "origin", "model_year"]).sum()

filt = df[["car name", "origin", "model_year"]].duplicated(keep = False)
df.loc[filt]

df.drop_duplicates(subset = ["car name", "origin", "model_year"], inplace = True)

# test
num_duplicates = df.duplicated(subset = ["car name", "origin", "model_year"]).sum()
assert num_duplicates == 0

num_cols = ['acceleration', 'cylinders', 'displacement', 'horsepower',
       'mpg', 'weight']
num_df = df[num_cols]

num_df.describe()

# I will draw boxplots to investigate outliers
fig, ax = plt.subplots(2, 3, figsize=(25, 25))

for i, col in enumerate(num_df.columns):
    g = sns.boxplot(data=df, y=col, ax=ax[i // 3, i % 3], palette="pastel")
    g.set_title(col, weight="bold", fontsize=18, fontname="monospace")

# Display the plot in Streamlit
st.pyplot(fig)

# let's see how many outliers are in every column
Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#condition
filt = ((num_df < (lower_bound)) | (num_df > (upper_bound)))
filt.sum()

df.loc[filt["acceleration"]]

df.loc[filt["horsepower"]]

df.loc[filt["mpg"]]

df.drop("car name", axis = 1, inplace = True)

def count_pie(data, col, hue = None, ax = None):
    
    # draw the count plot
    count =sns.countplot(data = df, x  = col, hue = hue ,palette = "pastel", ax = ax[0])
        # annotating
    for p in count.patches:
        count.annotate("{:.0f}".format(p.get_height()), (p.get_x() + 0.3, p.get_height() + 1), 
                       ha = "center", va = "bottom", color = "black", fontname = "monospace", fontsize = 15, weight = "bold")
    
    # draw pie plot
    labels = data[col].value_counts().sort_index().index
    data = data[col].value_counts().sort_index().values
    colors = sns.color_palette("pastel")[:len(labels)]
    _, labels_lst, percentage_labels_lst = ax[1].pie(data, labels = labels, colors = colors, autopct = "%0.0f%%",
              explode = [0.03 for i in range(len(labels))] )
    
    # labels
    
        # count plot
    count.set_xlabel(f"{col} Count plot", weight = "semibold", fontname = "monospace", fontsize = 15)
    count.set_ylabel("Count", weight = "semibold", fontname = "monospace", fontsize = 15)
    count.set_xticklabels(labels, fontsize = 15, weight = "bold")
    count.set_title(f"{col} count plot", weight = "bold", fontname = "monospace", fontsize = 25)
        # pie plot
    ax[1].set_xlabel(f"{col} Pie plot", weight = "semibold", fontname = "monospace", fontsize = 15)
    for label in labels_lst[:len(labels)]:
        label.update({"weight": "bold", "fontsize":15})
    for label in percentage_labels_lst[:len(labels)]:
        label.update({"weight": "bold", "fontsize":15})

    ax[1].legend(loc = "upper left", frameon = True, prop = {"size":15}, bbox_to_anchor = (1.05, 1))
    ax[1].set_title(f"{col} pie plot", weight = "bold", fontname = "monospace", fontsize = 25)

    

def scatter(df, x, y, ax = None, hue = None, size = None, style = None, alpha = 1):
    g = sns.scatterplot(data = df, x = x, y = y, ax  = ax , hue = hue,
                    size = size, style = style, markers = True, alpha = alpha, palette = "pastel")
    
    # titles
    g.set_title(f"{y} with {x}", fontsize = 17, weight = "bold", fontname = "monospace", pad = 20)
    g.set_xlabel(x, fontsize = 15, weight = "semibold", fontname = "monospace")
    g.set_ylabel(y, fontsize = 15, weight = "semibold", fontname = "monospace")

def violin(df, y, x = None, hue = None, ax = None):
    
    g = sns.violinplot(data = df, x = x, y = y, hue = hue, palette = "pastel", ax = ax)
    
    # titles
    g.set_title(f"{y} violin plot with {x}", fontsize = 17, weight = "bold", fontname = "monospace", pad = 20)
    g.set_xlabel(x, fontsize = 15, weight = "semibold", fontname = "monospace")
    g.set_ylabel(y, fontsize = 15, weight = "semibold", fontname = "monospace")

def plot_hist(df, col, hue = None, ax = None):
    plot = sns.histplot(data = df, x = col, kde= True, hue = hue, palette = "pastel", ax = ax)
        
    # titles
    plot.set_title(f"histogram plot for column {col}", fontsize = 15,weight = "bold", 
                fontname = "monospace", pad = 20)
    plot.set_xlabel(col, fontsize = 10,  weight = "semibold", fontname = "monospace")
    plot.set_ylabel("Count", fontsize = 10,  weight = "semibold", fontname = "monospace")    

st.write(df.corr())

# Crearea unui heatmap
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
# Afișarea heatmap-ului în Streamlit
st.pyplot(fig)

cat_cols = ["origin", "model_year", "cylinders"]
fig, ax = plt.subplots(3, 2, figsize = (20, 30))
for i, col in enumerate(cat_cols):
    count_pie(df, col, ax = (ax[i % 3, i//3], ax[i%3, i//3 + 1]))
st.pyplot(fig)

# Coloane numerice pentru histogramă
num_cols = ["acceleration", "horsepower", "weight", "displacement", "mpg"]
fig, ax = plt.subplots(2, 3, figsize=(20, 20))

# Plotăm histogramele pentru fiecare coloană numerică
for i, col in enumerate(num_cols):
    plot_hist(df, col, ax=ax[i//3, i%3])

# Înlăturăm celula goală
ax[1, 2].remove()

# Afișarea figurii în Streamlit
st.pyplot(fig)

# Coloane numerice pentru histogramă
num_cols = ["acceleration", "horsepower", "weight", "displacement", "mpg"]
fig, ax = plt.subplots(2, 3, figsize=(20, 20))

# Plotăm histogramele pentru fiecare coloană numerică cu hue
for i, col in enumerate(num_cols):
    plot_hist(df, col, ax=ax[i//3, i%3], hue="origin")

# Înlăturăm celula goală
ax[1, 2].remove()

# Afișarea figurii în Streamlit
st.pyplot(fig)

# Coloane numerice pentru plotarea scatter-urilor
num_cols = ["acceleration", "horsepower", "weight", "displacement"]
fig, ax = plt.subplots(2, 2, figsize=(20, 20))

# Creăm scatter plot-uri pentru fiecare coloană numerică din num_cols
for i, col in enumerate(num_cols):
    scatter(df, "mpg", col, ax=ax[i%2, i//2])

# Afișăm figura în Streamlit
st.pyplot(fig)

# Crearea graficului pairplot
sns.pairplot(df, vars=["horsepower", "acceleration", "displacement", "weight"])

# Afișarea graficului în Streamlit
st.pyplot(fig=plt.gcf()) 

# train-test split
X = df.drop("mpg", axis = 1)
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

for data in (X_train, X_test, y_train, y_test):
    print(data.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regs = [KNeighborsRegressor(), LinearRegression(),
              Lasso(), Ridge(), ElasticNet(), AdaBoostRegressor(), SVR(),
              RandomForestRegressor(), DecisionTreeRegressor(),
              ExtraTreesRegressor(), GradientBoostingRegressor()]

# Lista de modele de regresie
regs = [KNeighborsRegressor(), LinearRegression(),
        Lasso(), Ridge(), ElasticNet(), AdaBoostRegressor(), SVR(),
        RandomForestRegressor(), DecisionTreeRegressor(),
        ExtraTreesRegressor(), GradientBoostingRegressor()]

model_name = []
model_mae = []
model_r2 = []
model_mse = []

# Evaluarea fiecărui model
for reg in regs:
    # Cross-validation pentru predicții
    y_pred = cross_val_predict(reg, X_train, y_train, cv = 3)
    
    # Calcul MAE
    model_name.append(reg.__class__.__name__)
    model_mae.append(mae(y_train, y_pred))
    
    # Calcul R²
    r2 = r2_score(y_train, y_pred)
    model_r2.append(r2)
    
    # Calcul MSE
    model_mse.append(mse(y_train, y_pred))

# Crearea unui DataFrame cu rezultatele
final = pd.DataFrame({"name": model_name, "mae": model_mae, "r2": model_r2, "mse": model_mse})

# Sortare după MAE
final.sort_values(by = "mae", inplace = True)

# Afișare tabel
print(final)

# Antrenarea și evaluarea modelului LinearRegression pe setul de test
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calcul MAE, R² și MSE pentru modelul pe setul de test
test_mae = mae(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
test_mse = mse(y_test, y_pred)

# Afișare rezultate pentru modelul de test
st.write(f"Test MAE: {test_mae}")
st.write(f"Test R²: {test_r2}")
st.write(f"Test MSE: {test_mse}")




