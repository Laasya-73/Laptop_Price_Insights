import streamlit as st
from matplotlib import image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import re
import plotly.express as px
import os

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "images", "laptop.jpeg")
DATA_PATH = os.path.join(dir_of_interest, "data", "laptop_details.csv")

st.title("Dashboard - Laptop Price Prediction")

img = image.imread(IMAGE_PATH)
st.image(img)

df = pd.read_csv(DATA_PATH)


df['MRP'] = df['MRP'].str.replace('₹','').str.replace(',', '').astype(float)

df["Processor"] = df["Feature"].apply(lambda x: re.findall(r'(\w+\s*\w*) Processor', x)[0] if re.findall(r'(\w+\s*\w*) Processor', x) else "Unknown")
df["RAM Size"] = df["Feature"].apply(lambda x: re.findall(r'(\d+)\s*GB DDR\d', x)[0] if re.findall(r'(\d+)\s*GB DDR\d', x) else 0)
df["RAM Type"] = df["Feature"].apply(lambda x: re.findall(r'\d+ GB (DDR\d+)', x)[0] if re.findall(r'\d+ GB (DDR\d+)', x) else "Unknown")
df["Storage"] = df["Feature"].apply(lambda x: re.findall(r'\d+?\s*(?:TB|GB)\s*(?:HDD|SSD)', x)[0] if re.findall(r'\d+?\s*(?:TB|GB)\s*(?:HDD|SSD)', x) else "Unknown")
df["OS"] = df["Feature"].apply(lambda x: re.findall(r'(?:Windows|Mac|Linux|Ubuntu)\s+\d+', x)[0] if re.findall(r'(?:Windows|Mac|Linux|Ubuntu)\s+\d+', x) else "Unknown")

st.dataframe(df)

processor = st.selectbox("Processor: ",df['Processor'].unique())
ram_size = st.selectbox("RAM Size: ",df['RAM Size'].unique())
ram_type = st.selectbox("RAM Type: ",df['RAM Type'].unique())
storage = st.selectbox("HDD/SSD: ",df['Storage'].unique())
os = st.selectbox("OS: ",df['OS'].unique())

X = df[['Processor', 'RAM Size', 'RAM Type', 'Storage', 'OS']]
y = df['MRP']

# label encode the categorical variables
encoder = LabelEncoder()
X['Processor'] = encoder.fit_transform(X['Processor'])
X['RAM Type'] = encoder.fit_transform(X['RAM Type'])
X['Storage'] = encoder.fit_transform(X['Storage'])
X['OS'] = encoder.fit_transform(X['OS'])


# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# make predictions on test set
y_pred = rf.predict(X_test)

# Create a new dataframe with the selected features
new_data = pd.DataFrame({'Processor': [processor], 'RAM Size': [ram_size], 'RAM Type': [ram_type], 'Storage': [storage], 'OS': [os]})
new_data['Processor'] = encoder.fit_transform(new_data['Processor'])
new_data['RAM Type'] = encoder.fit_transform(new_data['RAM Type'])
new_data['Storage'] = encoder.fit_transform(new_data['Storage'])
new_data['OS'] = encoder.fit_transform(new_data['OS'])

# Make predictions on the new data using the trained model
predicted_price = rf.predict(new_data)

# Display the predicted price to the user
btn_click=st.button('Predict Price')
if btn_click == True:
   price=round(predicted_price[0])
   st.write("Predicted Price: ₹", price)

# data and insights for pricing strategies
st.subheader("Pricing Strategies")

# group laptops by processor and display average MRP
processor_prices = df.groupby('Processor')['MRP'].mean().reset_index()
fig1 = px.bar(processor_prices, x='Processor', y='MRP', title='Average MRP by Processor')
st.plotly_chart(fig1)

# group laptops by RAM size and display average MRP
ram_prices = df.groupby('RAM Size')['MRP'].mean().reset_index()
fig2 = px.bar(ram_prices, x='RAM Size', y='MRP', title='Average MRP by RAM Size')
st.plotly_chart(fig2)

# group laptops by RAM type and display average MRP
ramtype_prices = df.groupby('RAM Type')['MRP'].mean().reset_index()
fig3 = px.bar(ramtype_prices, x='RAM Type', y='MRP', title='Average MRP by RAM Type')
st.plotly_chart(fig3)

# group laptops by operating system and display average MRP
os_prices = df.groupby('OS')['MRP'].mean().reset_index()
fig4 = px.bar(os_prices, x='OS', y='MRP', title='Average MRP by Operating System')
st.plotly_chart(fig4)

# display pricing recommendations
st.write("Based on the data and insights from the above visualizations, the following recommendations can be made for pricing strategies for different types of laptops:")
st.write("1. Laptops with higher-end processors should be priced higher than those with lower-end processors.")
st.write("2. Laptops with larger RAM sizes should be priced higher than those with smaller RAM sizes.")
st.write("3. Laptops with DDR4 RAM should be priced higher than those with DDR3 RAM.")
st.write("4. Laptops with larger storage capacities should be priced higher than those with smaller storage capacities.")
st.write("5. Laptops with premium operating systems should be priced higher than those with basic operating systems.")





