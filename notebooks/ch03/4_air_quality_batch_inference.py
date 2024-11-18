#!/usr/bin/env python
# coding: utf-8

# # <span style="font-width:bold; font-size: 3rem; color:#1EB182;"> **Air Quality** </span><span style="font-width:bold; font-size: 3rem; color:#333;">- Part 04: Batch Inference</span>
# 
# ## 🗒️ This notebook is divided into the following sections:
# 
# 1. Download model and batch inference data
# 2. Make predictions, generate PNG for forecast
# 3. Store predictions in a monitoring feature group adn generate PNG for hindcast

# ## <span style='color:#ff5f27'> 📝 Imports

# In[1]:


import datetime
import pandas as pd
from xgboost import XGBRegressor
import hopsworks
import json
from functions import util
import os


# In[2]:


today = datetime.datetime.now() - datetime.timedelta(0)
tomorrow = today + datetime.timedelta(days = 1)
today


# ## <span style="color:#ff5f27;"> 📡 Connect to Hopsworks Feature Store </span>

# In[3]:


#with open('../../data/hopsworks-api-key.txt', 'r') as file:
#    os.environ["HOPSWORKS_API_KEY"] = file.read().rstrip()

project = hopsworks.login()
fs = project.get_feature_store() 

secrets = util.secrets_api(project.name)
location_str = secrets.get_secret("SENSOR_LOCATION_JSON").value
location = json.loads(location_str)
country=location['country']
city=location['city']
street=location['street']


# ## <span style="color:#ff5f27;"> ⚙️ Feature View Retrieval</span>
# 

# In[4]:


feature_view = fs.get_feature_view(
    name='air_quality_fv',
    version=1,
)


# ## <span style="color:#ff5f27;">🪝 Download the model from Model Registry</span>

# In[5]:


mr = project.get_model_registry()

retrieved_model = mr.get_model(
    name="air_quality_xgboost_model",
    version=1,
)

# Download the saved model artifacts to a local directory
saved_model_dir = retrieved_model.download()


# In[6]:


# Loading the XGBoost regressor model and label encoder from the saved model directory
# retrieved_xgboost_model = joblib.load(saved_model_dir + "/xgboost_regressor.pkl")
retrieved_xgboost_model = XGBRegressor()

retrieved_xgboost_model.load_model(saved_model_dir + "/model.json")

# Displaying the retrieved XGBoost regressor model
retrieved_xgboost_model


# ## <span style="color:#ff5f27;">✨ Get Weather Forecast Features with Feature View   </span>
# 
# 

# In[7]:


weather_fg = fs.get_feature_group(
    name='weather',
    version=1,
)
batch_data = weather_fg.filter(weather_fg.date >= today).read()
batch_data


# ### <span style="color:#ff5f27;">🤖 Making the predictions</span>

# In[8]:


batch_data['predicted_pm25'] = retrieved_xgboost_model.predict(
    batch_data[['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max', 'wind_direction_10m_dominant']])
batch_data


# In[9]:


batch_data.info()


# ### <span style="color:#ff5f27;">🤖 Saving the predictions (for monitoring) to a Feature Group</span>

# In[10]:


batch_data['street'] = street
batch_data['city'] = city
batch_data['country'] = country
# Fill in the number of days before the date on which you made the forecast (base_date)
batch_data['days_before_forecast_day'] = range(1, len(batch_data)+1)
batch_data = batch_data.sort_values(by=['date'])
batch_data


# In[11]:


batch_data.info()


# ### Create Forecast Graph
# Draw a graph of the predictions with dates as a PNG and save it to the github repo
# Show it on github pages

# In[12]:


file_path = "../../docs/air-quality/assets/img/pm25_forecast.png"
plt = util.plot_air_quality_forecast(city, street, batch_data, file_path)
plt.show()


# In[13]:


# Get or create feature group
monitor_fg = fs.get_or_create_feature_group(
    name='aq_predictions',
    description='Air Quality prediction monitoring',
    version=1,
    primary_key=['city','street','date','days_before_forecast_day'],
    event_time="date"
)


# In[14]:


monitor_fg.insert(batch_data, write_options={"wait_for_job": True})


# In[15]:


# We will create a hindcast chart for  only the forecasts made 1 day beforehand
monitoring_df = monitor_fg.filter(monitor_fg.days_before_forecast_day == 1).read()
monitoring_df


# In[16]:


air_quality_fg = fs.get_feature_group(
    name='air_quality',
    version=1,
)
air_quality_df = air_quality_fg.read()
air_quality_df


# In[17]:


outcome_df = air_quality_df[['date', 'pm25']]
preds_df =  monitoring_df[['date', 'predicted_pm25']]

hindcast_df = pd.merge(preds_df, outcome_df, on="date")
hindcast_df = hindcast_df.sort_values(by=['date'])

# If there are no outcomes for predictions yet, generate some predictions/outcomes from existing data
if len(hindcast_df) == 0:
    hindcast_df = util.backfill_predictions_for_monitoring(weather_fg, air_quality_df, monitor_fg, retrieved_xgboost_model)
hindcast_df


# ### Plot the Hindcast comparing predicted with forecasted values (1-day prior forecast)
# 
# __This graph will be empty to begin with - this is normal.__
# 
# After a few days of predictions and observations, you will get data points in this graph.

# In[18]:


file_path = "../../docs/air-quality/assets/img/pm25_hindcast_1day.png"
plt = util.plot_air_quality_forecast(city, street, hindcast_df, file_path, hindcast=True)
plt.show()


# ---
