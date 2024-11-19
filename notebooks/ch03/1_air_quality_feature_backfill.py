#!/usr/bin/env python
# coding: utf-8

# <span style="font-width:bold; font-size: 3rem; color:#333;">- Part 01: Feature Backfill for Air Quality Data</span>
# 
# 
# ## 🗒️ You have the following tasks
# 1. Choose an Air Quality Sensor
# 2. Update the country, city, and street information to point to YOUR chosen Air Quality Sensor
# 3. Download historical measures for your Air Quality Sensor as a CSV file
# 4. Update the path of the CSV file in this notebook to point to the one that you downloaded
# 5. Create an account on www.hopsworks.ai and get your HOPSWORKS_API_KEY
# 6. Run this notebook
# 
# 

# ### <span style='color:#ff5f27'> 📝 Imports

# In[1]:


import datetime
import requests
import pandas as pd
import hopsworks
import datetime
from pathlib import Path
from functions import util
import json
import re
import os
import warnings
warnings.filterwarnings("ignore")


# ### IF YOU WANT TO WIPE OUT ALL OF YOUR FEATURES AND MODELS, run the cell below

# In[2]:


# If you haven't set the env variable 'HOPSWORKS_API_KEY', then uncomment the next line and enter your API key
with open('../../data/hopsworks-api-key.txt', 'r') as file:
    os.environ["HOPSWORKS_API_KEY"] = file.read().rstrip()
#proj = hopsworks.login()
#util.purge_project(proj)


# ## <span style='color:#ff5f27'> 🌍 STEP 1: Pick your Air Quality Sensor</span>
# 
# ![image.png](attachment:b40d25b6-4994-4674-970b-a1eb4a14b9ad.png)
# 
#   * Find your favorite sensor on https://waqi.info/ 
#   * The sensor should have a URL in one of the two following forms: 
# 
#   `https://aqicn.org/station/<CITY OR COUNTRY NAME>/<STREET>`
#   or
# 
#   `https://aqicn.org/station/@36655//`
#   or 
#   
#   `https://aqicn.org/city/<CITY OR COUNTRY NAME>/<STREET>`
# 
# With your URL, we will need to do two things:
# 
#  * download the historical air quality readings as a CSV file
#  * note down the URL for the real-time API (you will need to create an API key for accessing this).
# 
# If your sensor's URL has one of the first two formats (the first URL path component is `station`), you will find the links to both historical CSV data and the real-time API on the same web page.
# 
# However, if your sensor's URL has the last format above (the first URL path component is `city` instead of `station`), then you will need to use 2 different URLs - one to download the historical CSV data and one for the real-time air quality measurements. You will find both of those links in the "Air quality historical data" section. Click on the "Historical air quality data" when you need to download the CSV file, and when you need the real-time API, click on the "Real-time air quality data".
# 
# Some examples of URLs for stations:
# 
#  * https://aqicn.org/station/sweden/stockholm-hornsgatan-108-gata/ - in Stockholm, Sweden
#  * https://aqicn.org/station/@36655// - in Hell's Kitchen, Manhatten, New York, USA
#  * https://aqicn.org/station/nigeria-benin-city-edo-state-secretariat-palm-house/ - in Benin City, Nigeria
#  * https://aqicn.org/station/india/mumbai/sion/ - Sion in Mumbai, India
# 
# Here is what the webpage at URL for the Stockholm sensor looks like:
# 
# ![station.png](attachment:1922302e-13b9-469d-ba75-63dabf7b4475.png)
# 
# __When you pick a sensor for your project, there are 2 things the sensor MUST have__:
#   1. __PM 2.5__ measurements
#   2. __Good Historical Values__ for download as a CSV file
# 
# __Write down the country, city, and the street for your sensor.__
# 
# We will use the city name to download weather data for your sensor, and we will store the country and street values in the sensor's feature group.

# ## What makes a good quality Air Quality Sensor?
# 
# In the image below, we can see below that our sensor in Stockholm fulfills our 2 requirements. It has:
#   1. __PM 2.5__ measurements (see upper red-ringed value in image below)
#   2. __Good Historical Measurements__ with few missing values (see lower red-ringed values in image below) 
# 
# ![sensor.png](attachment:46ebde65-85ff-4b12-a560-65230881db0b.png)

# ---

# ## <span style='color:#ff5f27'> 🌍 STEP 2: Download the Historical Air Quality </span>
# 
# You can download a CSV file containing the historical air quality data from your your sensor's URL.
# Scroll down to the section "Air Quality Historical Data". Click on the PM2.5 option and save the file to the `data` directory in your forked version of this Github repository. Note the name of your CSV file, you will need 
# 
# 
# ![download-csv.png](attachment:ab17240a-17ad-47de-97af-2a3a1d32f247.png)

# ## <span style='color:#ff5f27'> 🌍 STEP 3: Get the AQICN_URL and API key. Enter country, city, street names for your Sensor.</span>
# 
# You can find your __AQICN_URL__ if you scroll down the webpage for your sensor - it is the URL inside the redbox here.
# You shouldn't include the last part of the url - "/?token=\__YOUR_TOKEN\__". 
# It is bad practice to save TOKENs (aka API KEYs) in your source code - you might make it public if you check that code into Github!
# We will fill in the token later by saving the AQI_API_KEY as a secret in Hopsworks.
# 
# ![stockholm-rt-api.png](attachment:70fea299-d303-49f8-99ba-43e981d7c3aa.png)
# 
# __Update the values in the cell below.__

# In[3]:


csv_file="../../data/aristotelous-air-quality.csv"
util.check_file_path(csv_file)


# In[4]:


# TODO: Change these values to point to your Sensor
country="greece"
city = "athens"
street = "aristotelous"
aqicn_url="https://api.waqi.info/feed/@12414"

# This API call may fail if the IP address you run this notebook from is blocked by the Nominatim API
# If this fails, lookup the latitude, longitude using Google and enter the values here.
latitude, longitude = util.get_city_coordinates(city)

today = datetime.date.today()


# ## <span style='color:#ff5f27'> 🌍 STEP 4: Get an AQI API Token and Store it as a Secret in Hopsworks </span>
# 
# You have to first get your AQI API key [from here](https://aqicn.org/data-platform/token/):
# 
# ![image.png](attachment:9336a7d3-f7dc-4aec-a854-78e787e4493e.png)
# 
# 
# Save the API KEY to a new file you need to create in your repository (do not check this file into Github!):
# 
#  * <project root>/data/aqi-api-key.txt

# In[5]:


aqi_api_key_file = '../../data/aqi-api-key.txt'
util.check_file_path(aqi_api_key_file)

with open(aqi_api_key_file, 'r') as file:
    AQI_API_KEY = file.read().rstrip()


# ## Hopsworks API Key
# You need to have registered an account on app.hopsworks.ai.
# You will be prompted to enter your API key here, unless you set it as the environment variable HOPSWORKS_API_KEY (my preffered approach).

# In[6]:


with open('../../data/hopsworks-api-key.txt', 'r') as file:
    os.environ["HOPSWORKS_API_KEY"] = file.read().rstrip()
    
project = hopsworks.login()


# In[7]:


secrets = util.secrets_api(project.name)
try:
    secrets.create_secret("AQI_API_KEY", AQI_API_KEY)
except hopsworks.RestAPIError:
    AQI_API_KEY = secrets.get_secret("AQI_API_KEY").value


# ### Validate that the AQI_API_KEY you added earlier works
# 
# The cell below should print out something like:
# 
# ![image.png](attachment:832cc3e9-876c-450f-99d3-cc97abb55b13.png)

# In[8]:


try:
    aq_today_df = util.get_pm25(aqicn_url, country, city, street, today, AQI_API_KEY)
except hopsworks.RestAPIError:
    print("It looks like the AQI_API_KEY doesn't work for your sensor. Is the API key correct? Is the sensor URL correct?")

aq_today_df.head()


# ## <span style='color:#ff5f27'> 🌍 STEP 5: Read your CSV file into a DataFrame </span>
# 
# The cell below will read up historical air quality data as a CSV file into a Pandas DataFrame

# In[9]:


df = pd.read_csv(csv_file,  parse_dates=['date'], skipinitialspace=True)
df


# ## <span style='color:#ff5f27'> 🌍 STEP 6: Data cleaning</span>
# 
# 
# ### Rename columns if needed and drop unneccessary columns
# 
# We want to have a DataFrame with 2 columns - `date` and `pm25` after this cell below:

# ## Check the data types for the columns in your DataFrame
# 
#  * `date` should be of type   datetime64[ns] 
#  * `pm25` should be of type float64

# In[10]:


# These commands will succeed if your CSV file didn't have a `median` or `timestamp` column
df = df.rename(columns={"median": "pm25"})
df = df.rename(columns={"timestamp": "date"})

df_aq = df[['date', 'pm25']]
df_aq['pm25'] = df_aq['pm25'].astype('float32')

df_aq


# In[11]:


# Cast the pm25 column to be a float32 data type
df_aq.info()


# ## <span style='color:#ff5f27'> 🌍 STEP 7: Drop any rows with missing data </span>
# It will make the model training easier if there is no missing data in the rows, so we drop any rows with missing data.

# In[12]:


df_aq.dropna(inplace=True)
df_aq


# ## <span style='color:#ff5f27'> 🌍 STEP 8: Add country, city, street, url to the DataFrame </span>
# 
# Your CSV file may have many other air quality measurement columns. We will only work with the `pm25` column.
# 
# We add the columns for the country, city, and street names that you changed for your Air Quality sensor.
# 
# We also want to make sure the `pm25` column is a float32 data type.

# In[13]:


# Your sensor may have columns we won't use, so only keep the date and pm25 columns
# If the column names in your DataFrame are different, rename your columns to `date` and `pm25`
df_aq['country']=country
df_aq['city']=city
df_aq['street']=street
df_aq['url']=aqicn_url
df_aq


# ---

# ## <span style='color:#ff5f27'> 🌦 Loading Weather Data from [Open Meteo](https://open-meteo.com/en/docs)

# ## <span style='color:#ff5f27'> 🌍 STEP 9: Download the Historical Weather Data </span>
# 
# https://open-meteo.com/en/docs/historical-weather-api#hourly=&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant
# 
# We will download the historical weather data for your `city` by first extracting the earliest date from your DataFrame containing the historical air quality measurements.
# 
# We will download all daily historical weather data measurements for your `city` from the earliest date in your air quality measurement DataFrame. It doesn't matter if there are missing days of air quality measurements. We can store all of the daily weather measurements, and when we build our training dataset, we will join up the air quality measurements for a given day to its weather features for that day. 
# 
# The weather features we will download are:
# 
#  * `temperature (average over the day)`
#  * `precipitation (the total over the day)`
#  * `wind speed (average over the day)`
#  * `wind direction (the most dominant direction over the day)`
# 

# In[14]:


earliest_aq_date = pd.Series.min(df_aq['date'])
earliest_aq_date = earliest_aq_date.strftime('%Y-%m-%d')
earliest_aq_date

weather_df = util.get_historical_weather(city, earliest_aq_date, str(today), latitude, longitude)


# In[15]:


weather_df.info()


# ## <span style='color:#ff5f27'> 🌍 STEP 10: Define Data Validation Rules </span>
# 
# We will validate the air quality measurements (`pm25` values) before we write them to Hopsworks.
# 
# We define a data validation rule (an expectation in Great Expectations) that ensures that `pm25` values are not negative or above the max value available by the sensor.
# 
# We will attach this expectation to the air quality feature group, so that we validate the `pm25` data every time we write a DataFrame to the feature group. We want to prevent garbage-in, garbage-out.

# In[16]:


import great_expectations as ge
aq_expectation_suite = ge.core.ExpectationSuite(
    expectation_suite_name="aq_expectation_suite"
)

aq_expectation_suite.add_expectation(
    ge.core.ExpectationConfiguration(
        expectation_type="expect_column_min_to_be_between",
        kwargs={
            "column":"pm25",
            "min_value":-0.1,
            "max_value":500.0,
            "strict_min":True
        }
    )
)


# ## Expectations for Weather Data
# Here, we define an expectation for 2 columns in our weather DataFrame - `precipitation_sum` and `wind_speed_10m_max`, where we expect both values to be greater than zero, but less than 1000.

# In[17]:


import great_expectations as ge
weather_expectation_suite = ge.core.ExpectationSuite(
    expectation_suite_name="weather_expectation_suite"
)

def expect_greater_than_zero(col):
    weather_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={
                "column":col,
                "min_value":-0.1,
                "max_value":1000.0,
                "strict_min":True
            }
        )
    )
expect_greater_than_zero("precipitation_sum")
expect_greater_than_zero("wind_speed_10m_max")


# ---

# ### <span style="color:#ff5f27;"> 🔮 STEP 11: Connect to Hopsworks and save the sensor country, city, street names as a secret</span>

# In[18]:


fs = project.get_feature_store() 


# #### Save country, city, street names as a secret
# 
# These will be downloaded from Hopsworks later in the (1) daily feature pipeline and (2) the daily batch inference pipeline

# In[19]:


dict_obj = {
    "country": country,
    "city": city,
    "street": street,
    "aqicn_url": aqicn_url,
    "latitude": latitude,
    "longitude": longitude
}

# Convert the dictionary to a JSON string
str_dict = json.dumps(dict_obj)

try:
    secrets.create_secret("SENSOR_LOCATION_JSON", str_dict)
except hopsworks.RestAPIError:
    print("SENSOR_LOCATION_JSON already exists. To update, delete the secret in the UI (https://c.app.hopsworks.ai/account/secrets) and re-run this cell.")
    existing_key = secrets.get_secret("SENSOR_LOCATION_JSON").value
    print(f"{existing_key}")


# ### <span style="color:#ff5f27;"> 🔮 STEP 12: Create the Feature Groups and insert the DataFrames in them </span>

# ### <span style='color:#ff5f27'> 🌫 Air Quality Data
#     
#  1. Provide a name, description, and version for the feature group.
#  2. Define the `primary_key`: we have to select which columns uniquely identify each row in the DataFrame - by providing them as the `primary_key`. Here, each air quality sensor measurement is uniquely identified by `country`, `street`, and  `date`.
#  3. Define the `event_time`: We also define which column stores the timestamp or date for the row - `date`.
#  4. Attach any `expectation_suite` containing data validation rules

# In[20]:


air_quality_fg = fs.get_or_create_feature_group(
    name='air_quality',
    description='Air Quality characteristics of each day',
    version=1,
    primary_key=['city', 'street', 'date'],
    event_time="date",
    expectation_suite=aq_expectation_suite
)


# #### Insert the DataFrame into the Feature Group

# In[21]:


air_quality_fg.insert(df_aq)


# #### Enter a description for each feature in the Feature Group

# In[22]:


air_quality_fg.update_feature_description("date", "Date of measurement of air quality")
air_quality_fg.update_feature_description("country", "Country where the air quality was measured (sometimes a city in acqcn.org)")
air_quality_fg.update_feature_description("city", "City where the air quality was measured")
air_quality_fg.update_feature_description("street", "Street in the city where the air quality was measured")
air_quality_fg.update_feature_description("pm25", "Particles less than 2.5 micrometers in diameter (fine particles) pose health risk")


# ### <span style='color:#ff5f27'> 🌦 Weather Data
#     
#  1. Provide a name, description, and version for the feature group.
#  2. Define the `primary_key`: we have to select which columns uniquely identify each row in the DataFrame - by providing them as the `primary_key`. Here, each weather measurement is uniquely identified by `city` and  `date`.
#  3. Define the `event_time`: We also define which column stores the timestamp or date for the row - `date`.
#  4. Attach any `expectation_suite` containing data validation rules

# In[23]:


# Get or create feature group 
weather_fg = fs.get_or_create_feature_group(
    name='weather',
    description='Weather characteristics of each day',
    version=1,
    primary_key=['city', 'date'],
    event_time="date",
    expectation_suite=weather_expectation_suite
) 


# #### Insert the DataFrame into the Feature Group

# In[24]:


# Insert data
weather_fg.insert(weather_df)


# #### Enter a description for each feature in the Feature Group

# In[25]:


weather_fg.update_feature_description("date", "Date of measurement of weather")
weather_fg.update_feature_description("city", "City where weather is measured/forecast for")
weather_fg.update_feature_description("temperature_2m_mean", "Temperature in Celsius")
weather_fg.update_feature_description("precipitation_sum", "Precipitation (rain/snow) in mm")
weather_fg.update_feature_description("wind_speed_10m_max", "Wind speed at 10m abouve ground")
weather_fg.update_feature_description("wind_direction_10m_dominant", "Dominant Wind direction over the dayd")


# ## <span style="color:#ff5f27;">⏭️ **Next:** Part 02: Daily Feature Pipeline 
#  </span> 
# 

# ## <span style="color:#ff5f27;">⏭️ **Exercises:** 
#  </span> 
# 
# Extra Homework:
# 
#   * Try adding a new feature based on a rolling window of 3 days for 'pm25'
#       * This is not easy, as forecasting more than 1 day in the future, you won't have the previous 3 days of pm25 measurements.
#       * df.set_index("date").rolling(3).mean() is only the start....
#   * Parameterize the notebook, so that you can provide the `country`/`street`/`city`/`url`/`csv_file` as parameters. 
#       * Hint: this will also require making the secret name (`SENSOR_LOCATION_JSON`), e.g., add the street name as part of the secret name. Then you have to pass that secret name as a parameter when running the operational feature pipeline and batch inference pipelines.
#       * After you have done this, collect the street/city/url/csv files for all the sensors in your city or region and you make dashboards for all of the air quality sensors in your city/region. You could even then add a dashboard for your city/region, as done [here for Poland](https://github.com/erno98/ID2223).
# 
# Improve this AI System
#   * As of mid 2024, there is no API call available to download historical data from the AQIN website. You could improve this system by writing a PR to download the CSV file using Python Selenium and the URL for the sensor.
# 

# ---
