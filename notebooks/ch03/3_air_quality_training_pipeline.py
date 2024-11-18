#!/usr/bin/env python
# coding: utf-8

# # <span style="font-width:bold; font-size: 3rem; color:#333;">Training Pipeline</span>
# 
# ## 🗒️ This notebook is divided into the following sections:
# 
# 1. Select features for the model and create a Feature View with the selected features
# 2. Create training data using the feature view
# 3. Train model
# 4. Evaluate model performance
# 5. Save model to model registry

# ### <span style='color:#ff5f27'> 📝 Imports

# In[1]:


import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, r2_score
import hopsworks
from functions import util

import warnings
warnings.filterwarnings("ignore")


# ## <span style="color:#ff5f27;"> 📡 Connect to Hopsworks Feature Store </span>

# In[2]:


# If you haven't set the env variable 'HOPSWORKS_API_KEY', then uncomment the next line and enter your API key
#with open('../../data/hopsworks-api-key.txt', 'r') as file:
#    os.environ["HOPSWORKS_API_KEY"] = file.read().rstrip()

project = hopsworks.login()
fs = project.get_feature_store() 


# In[3]:


# Retrieve feature groups
air_quality_fg = fs.get_feature_group(
    name='air_quality',
    version=1,
)
weather_fg = fs.get_feature_group(
    name='weather',
    version=1,
)


# --- 
# 
# ## <span style="color:#ff5f27;"> 🖍 Feature View Creation and Retrieving </span>

# In[4]:


# Select features for training data.
selected_features = air_quality_fg.select(['pm25']).join(weather_fg.select_all(), on=['city'])
selected_features.show(10)


# ### Feature Views
# 
# `Feature Views` are selections of features from different **Feature Groups** that make up the input and output API (or schema) for a model. A **Feature Views** can create **Training Data** and also be used in Inference to retrieve inference data.
# 
# The Feature Views allows a schema in form of a query with filters, defining a model target feature/label and additional transformation functions (declarative feature encoding).
# 
# In order to create Feature View we can use `FeatureStore.get_or_create_feature_view()` method.
# 
# You can specify the following parameters:
# 
# - `name` - name of a feature group.
# 
# - `version` - version of a feature group.
# 
# - `labels`- our target variable.
# 
# - `transformation_functions` - declarative feature encoding (not used here)
# 
# - `query` - selected features/labels for the model 

# In[5]:


feature_view = fs.get_or_create_feature_view(
    name='air_quality_fv',
    description="weather features with air quality as the target",
    version=1,
    labels=['pm25'],
    query=selected_features,
)


# ## <span style="color:#ff5f27;">🪝 Split the training data into train/test data sets </span>
# 
# We use a time-series split here, with training data before this date `start_date_test_data` and test data after this date

# In[6]:


start_date_test_data = "2024-03-01"
# Convert string to datetime object
test_start = datetime.strptime(start_date_test_data, "%Y-%m-%d")


# In[7]:


X_train, X_test, y_train, y_test = feature_view.train_test_split(
    test_start=test_start
)


# In[8]:


X_train


# In[9]:


# Drop the index columns - 'date' (event_time) and 'city' (primary key)

train_features = X_train.drop(['date', 'city'], axis=1)
test_features = X_test.drop(['date', 'city'], axis=1)


# In[10]:


y_train


# The `Feature View` is now saved in Hopsworks and you can retrieve it using `FeatureStore.get_feature_view(name='...', version=1)`.

# ---

# ## <span style="color:#ff5f27;">🧬 Modeling</span>
# 
# We will train a regression model to predict pm25 using our 4 features (wind_speed, wind_dir, temp, precipitation)

# In[11]:


# Creating an instance of the XGBoost Regressor
xgb_regressor = XGBRegressor()

# Fitting the XGBoost Regressor to the training data
xgb_regressor.fit(train_features, y_train)


# In[12]:


# Predicting target values on the test set
y_pred = xgb_regressor.predict(test_features)

# Calculating Mean Squared Error (MSE) using sklearn
mse = mean_squared_error(y_test.iloc[:,0], y_pred)
print("MSE:", mse)

# Calculating R squared using sklearn
r2 = r2_score(y_test.iloc[:,0], y_pred)
print("R squared:", r2)


# In[13]:


df = y_test
df['predicted_pm25'] = y_pred


# In[14]:


df['date'] = X_test['date']
df = df.sort_values(by=['date'])
df.head(5)


# In[15]:


# Creating a directory for the model artifacts if it doesn't exist
model_dir = "air_quality_model"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
images_dir = model_dir + "/images"
if not os.path.exists(images_dir):
    os.mkdir(images_dir)


# In[16]:


file_path = images_dir + "/pm25_hindcast.png"
plt = util.plot_air_quality_forecast("athens", "aristotelous-air-quality", df, file_path, hindcast=True) 
plt.show()


# In[17]:


# Plotting feature importances using the plot_importance function from XGBoost
plot_importance(xgb_regressor, max_num_features=4)
feature_importance_path = images_dir + "/feature_importance.png"
plt.savefig(feature_importance_path)
plt.show()


# ---

# ## <span style='color:#ff5f27'>🗄 Model Registry</span>
# 
# One of the features in Hopsworks is the model registry. This is where you can store different versions of models and compare their performance. Models from the registry can then be served as API endpoints.

# ### <span style="color:#ff5f27;">⚙️ Model Schema</span>

# The model needs to be set up with a [Model Schema](https://docs.hopsworks.ai/machine-learning-api/latest/generated/model_schema/), which describes the inputs and outputs for a model.
# 
# A Model Schema can be automatically generated from training examples, as shown below.

# In[18]:


from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# Creating input and output schemas using the 'Schema' class for features (X) and target variable (y)
input_schema = Schema(X_train)
output_schema = Schema(y_train)

# Creating a model schema using 'ModelSchema' with the input and output schemas
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

# Converting the model schema to a dictionary representation
schema_dict = model_schema.to_dict()


# In[19]:


# Saving the XGBoost regressor object as a json file in the model directory
xgb_regressor.save_model(model_dir + "/model.json")


# In[20]:


res_dict = { 
        "MSE": str(mse),
        "R squared": str(r2),
    }


# In[21]:


mr = project.get_model_registry()

# Creating a Python model in the model registry named 'air_quality_xgboost_model'

aq_model = mr.python.create_model(
    name="air_quality_xgboost_model", 
    metrics= res_dict,
    model_schema=model_schema,
    input_example=X_test.sample().values, 
    description="Air Quality (PM2.5) predictor",
)

# Saving the model artifacts to the 'air_quality_model' directory in the model registry
aq_model.save(model_dir)


# ---
# ## <span style="color:#ff5f27;">⏭️ **Next:** Part 04: Batch Inference</span>
# 
# In the following notebook you will use your model for Batch Inference.
# 

# In[ ]:




