import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pickle
import json

# Load the CSV file into pandas DataFrma
df = pd.read_csv("Bengaluru_House_Data.csv")
print(df, "\n")

# I will drop the column below, by assuming they wont affect the price of my model
houses_data = df.drop(["area_type", "society", "balcony", "availability"], axis=1)
print(houses_data, "\n")

"""I will now clean my data"""
#Checking for NaN
nan_value = houses_data.isna().sum()
print(nan_value, "\n")

# My name is insignificant therefore i will drop them
houses_data.dropna(inplace=True)
print(houses_data, "\n")

""" It can be seen from the size column that BHK and Bedrooms are basically the same things, I will therefore keep 
 the number in front of the words and drop the rest"""
# First i will view the format of this words in size column
print(houses_data["size"].unique(), "\n")
"""['2 BHK' '4 Bedroom' '3 BHK' '4 BHK' '6 Bedroom' '3 Bedroom' '1 BHK'
 '1 RK' '1 Bedroom' '8 Bedroom' '2 Bedroom' '7 Bedroom' '5 BHK' '7 BHK'
 '6 BHK' '5 Bedroom' '11 BHK' '9 BHK' '9 Bedroom' '27 BHK' '10 Bedroom'
 '11 Bedroom' '10 BHK' '19 BHK' '16 BHK' '43 Bedroom' '14 BHK' '8 BHK'
 '12 Bedroom' '13 BHK' '18 Bedroom']"""

""" As seen above, i will apply function to take the number part and drop the rest either BHK or Bedrooms and put the 
in a new column call BHK_BEDROOMS"""
houses_data["BHK_BEDROOM"] = houses_data["size"].apply(lambda x: int(x.split(" ")[0]))
print(houses_data, "\n")

# I will view the format in my new created column BHK_BEDROOM
print(houses_data["BHK_BEDROOM"].unique(), "\n")
"""[ 2  4  3  6  1  8  7  5 11  9 27 10 19 16 43 14 12 13 18], I can see some houses have a lot of bedrooms, i want to
know their square feet. I will therefore all print number of bedrooms greater than 12"""
size_check = houses_data[houses_data["BHK_BEDROOM"] > 12]
print(size_check)
""" I can see 2400 sqft with 43 bedrooms which is not possible, i will before need to further clean the data. To do data
cleaning of the rest of the errors in bedrooms square feet relations, i have to first check the format of total_sqft"""
print(houses_data["total_sqft"].unique(), "\n")
"""['1056' '2600' '1440' ... '1133 - 1384' '774' '4689'], from this it can be seen that there are some square feet given
in terms of range, to correct this i will need to find their average"""

"""I will write a functions to convert all the square feet numbers into float type, it the function is not able to 
convert then it means it a range of number with dash. Hence I will split and convert and take the average of both numbers 
"""
# Function to convert to float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
# Now I will apply this function to my total_sqft
range_sqft = houses_data[~houses_data["total_sqft"].apply(is_float)]
# print(is_float(range_sqft))
print(range_sqft)
"""Now that i have catch all the range values and values with different metric systems, will write a function to take
average of that of range values"""

def sqft_average(x):
    feet = x.split("-")
    if len(feet) == 2:
        return (float(feet[0])+float(feet[1])) / 2
    try:
        return float(x)
    except:
        return None

houses_data["total_sqft"] = houses_data["total_sqft"].apply(sqft_average)
print(houses_data)

"""Now I will do Feature Engineering on the data"""
# It important to know price pre sqft, hence i will create a column for that
houses_data["price_per_sqft"] = houses_data["price"]*100000/houses_data["total_sqft"]
print(houses_data, "\n")

"""I want to convert my location column using get dummies from pandas, first i have to check the number of locations
in my data"""
print(houses_data.location.unique())
print(len(houses_data.location.unique()))
"""['Electronic City Phase II' 'Chikka Tirupathi' 'Uttarahalli' ...
 '12th cross srinivas nagar banshankari 3rd stage' 'Havanur extension'
 'Abshot Layout'] these are the unique location and the count is 1304, the count is really a big number therefore i have 
 to reduce the location"""

# Removing space in front and behind the various locations
houses_data.location = houses_data.location.apply(lambda x: x.strip())

# Now i will group all the location into a variable
locations_stat = houses_data.groupby("location")["location"].agg("count").sort_values(ascending=False)
print(locations_stat)

"""I will put all locations with row less or equal to 10 into a category call other locations"""
location_less_than_10 = locations_stat[locations_stat<=10]
print(location_less_than_10)

# Creating other category
houses_data.location = houses_data.location.apply(lambda x: "other" if x in location_less_than_10 else x)
print(houses_data)

"""Removing Outlier"""
# removing which have less 300 sqft
houses_data = houses_data[~(houses_data.total_sqft/houses_data.BHK_BEDROOM<300)]
print(houses_data)

# Checking the statistics of price per sqft
print(houses_data)

# Create a function to filter the price per location, using mean and standard deviation
def remove_outliers(dataframe):
    filter_dataframe = pd.DataFrame()
    for key, subdf in dataframe.groupby("location"):
        means = np.mean(subdf.price_per_sqft)
        stds = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(means-stds)) & (subdf.price_per_sqft<=(means+stds))]
        filter_dataframe = pd.concat([filter_dataframe, reduced_df], ignore_index=True)
    return filter_dataframe
# Apply the function to the house data
houses_data = remove_outliers(houses_data)
print(houses_data)

"""In the data, it is seen that some houses with less number of rooms have high price than houses with more rooms, to 
have a view of this, i will write a function to view this behaviour on scatter plot"""
# Scatter plot function
def scatter_plot(dataframe, location):
    bhk2 = dataframe[(dataframe.location==location) & (dataframe.BHK_BEDROOM==2)]
    bhk3 = dataframe[(dataframe.location == location) & (dataframe.BHK_BEDROOM == 3)]
    plt.scatter(bhk2.total_sqft, bhk2.price, color="blue", label="2 BHK")
    plt.scatter(bhk3.total_sqft, bhk3.price, color="green", marker="+", label="3 BHK")
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
    plt.show()
scatter_plot(houses_data, "Hebbal")

# Now remove bedrooms outliers
def remove_bhk_outliers(dataframe):
    exclude_indices = np.array([])
    for location, location_df in dataframe.groupby("location"):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("BHK_BEDROOM"):
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby("BHK_BEDROOM"):
            stats = bhk_stats.get(bhk-1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft <(stats["mean"])].index.values)
    return dataframe.drop(exclude_indices, axis="index")

houses_data = remove_bhk_outliers(houses_data)
print(houses_data)
"""From this graph you can see that i have removed alot of outliers"""
scatter_plot(houses_data, "Hebbal")

"""NOw i will like to view the price per square feet area on histogram to see how many houses are in price per square 
feet area"""
plt.hist(houses_data.price_per_sqft, rwidth=0.5)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()
"""It can be seen that i have majority of my data from 0 to 10000"""

"""Now I will work on the bathrooms, first i will find out how many bathdrooms are in the houses, i will try to see houses
with more than 10 bathrooms and see if it make sense compare to the number of rooms and square feet area"""
print(houses_data[houses_data.bath > 10])

"""Now i will remove all houses with bathrooms more than number of rooms"""
# I will view relationship of bathrooms with houses on histogram
plt.hist(houses_data.bath, rwidth=0.5)
plt.xlabel("Number of Bathrooms")
plt.ylabel("Count")
plt.show()
"""It can be seen that most of the houses have 2, 4, 6 bathrooms with few with more than 8, 10, 12 bathrooms.
I will remove all houses with 2 more bathrooms than bedrooms"""
# First have a look at these houses
print(houses_data[houses_data.bath > houses_data.BHK_BEDROOM + 2])

# Now, removing these outliers
houses_data = houses_data[houses_data.bath < houses_data.BHK_BEDROOM + 2]
print(houses_data)

"""Now my data is ready for training, but i will drop size and price_per_sqft"""
houses_data = houses_data.drop(["size", "price_per_sqft"], axis=1)
print(houses_data)

"""From my data, i have to convert all my non numbers to numbers using get dummies from python"""
#Converting my locations to numbers
loc_number = pd.get_dummies(houses_data.location)
print(loc_number)

# I will concatenate loc_number to my house data, with "other" dropped
houses_data = pd.concat([houses_data, loc_number.drop("other", axis=1)], axis=1)
print(houses_data)

# I will drop the location column now
houses_data = houses_data.drop("location", axis=1)
print(houses_data)

# Creating x variables(In-dependable variables) and y variable(dependable variable)
x = houses_data.drop("price", axis=1)
y = houses_data.price
print(x)
print(y)

# Train my data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# Use linear regression
Reg = LinearRegression()
Reg.fit(x_train, y_train)

# Check the score of the model
Reg_score = Reg.score(x_test, y_test)
print(Reg_score)

# Using KFOLD cross validation to get the best score
cross_val = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
k_fold_score = cross_val_score(LinearRegression(), x, y, cv=cross_val)
print(k_fold_score)
"""my scores are: [0.82430186 0.77166234 0.85089567 0.80837764 0.83653286], i still need to search for the best model
therefore i will use GridSearchCV to find best model"""

# I will write a function to give me the best algorithm, hyper parameter tuning
def find_best_model_with_gridsearchcv(x, y):
    model_param = {
        "linear_regression": {
            "model": LinearRegression(),
            "params": {
                "normalize": [True, False],

            }
        },

        "lasso": {
            "model": Lasso(),
            "params": {
                "alpha": [1, 2],
                "selection": ["random", "cyclic"]

                }

            },
        "decision_tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "criterion": ["mse", "friedman_mse"],
                "splitter": ["best", "random"]

            }

        }
    }

    """Iterate through json object, use the GridSearchCV to select the best score, model and parameterd"""
    scores = []
    cross_val = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for model_name, config in model_param.items():
        grid_search = GridSearchCV(config["model"], config["params"], cv=cross_val, return_train_score=False)
        grid_search.fit(x, y)
        scores.append({
            "model": model_name,
            "best_score": grid_search.best_score_,
            "best_params": grid_search.best_params_
        })
    # Convert the scores list into pandas DataFrame
    return pd.DataFrame(scores)
best_score = find_best_model_with_gridsearchcv(x, y)
print(best_score)
"""               model  best_score                                 best_params
0  linear_regression    0.818354                         {'normalize': True}
1              lasso    0.687429         {'alpha': 2, 'selection': 'random'}
2      decision_tree    0.727739  {'criterion': 'mse', 'splitter': 'random'}
It can be seen that my linear regression model gives me the best model, therefore i will use it"""

"""Now, i will write a function to predict the price of a house base on location, sqft, bath, bhk"""
# print(x.columns)
def predict_price(location, sqft, bath, bhk):
    x = houses_data.drop("price", axis=1)
    loc_index = np.where(x.columns==location)[0][0]

    x = np.zeros(len(x.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return Reg.predict([x])[0]
predicted_price = predict_price("Electronic City Phase II", 1000, 3,3)
print(predicted_price)

"""Now my model is ready to be exported to pickle file and the x columns into json file"""
with open("home_prediction_model.pickle", "wb") as file:
    pickle.dump(Reg, file)

# Json file of my x columnx
columns = {
    "data_columns": [col.lower() for col in x.columns]
}
with open("columns.json", "w+") as file:
    file.write(json.dumps(columns))