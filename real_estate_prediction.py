import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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