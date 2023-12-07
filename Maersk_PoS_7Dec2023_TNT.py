#!/usr/bin/env python
# coding: utf-8

# In[38]:


# Required Libraries
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[2]:


# Load the data
data_raw = pd.read_csv("/Users/pikpes/Downloads/Maersk Data_General.csv)


# ## Internal Indicators

# In[3]:


# Visual Graph of Internal Indicators

import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

# Assume 'data_raw' is your DataFrame loaded with the raw data

# Function to process and normalize the data
def process_and_normalize_data(data, row_index, column_name):
    processed_data = data.iloc[row_index, 1:].T.reset_index()
    processed_data.columns = ['Year', column_name]
    processed_data[column_name] = (
        processed_data[column_name]
        .replace('—', pd.NA)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    processed_data[column_name] = pd.to_numeric(
        processed_data[column_name], errors='coerce'
    )
    # Normalize the data
    scaler = MinMaxScaler()
    processed_data[column_name] = scaler.fit_transform(
        processed_data[[column_name]].dropna())
    return processed_data

# Process and normalize the data for each variable
sulphur_dioxide_emissions_data = process_and_normalize_data(data_raw, 12, 'Sulphur Dioxide / Sulphur Oxide Emissions')
nitrogen_oxide_emissions_data = process_and_normalize_data(data_raw, 26, 'Nitrogen Oxide Emissions')
GHG_Scope_1_data = process_and_normalize_data(data_raw, 19, 'GHG Scope 1')

# Create scatter plots for each of these variables
def create_scatter_trace(data, name, color):
    return go.Scatter(
        x=data['Year'],
        y=data[name],
        mode='lines+markers',
        name=name,
        marker=dict(color=color),
        line=dict(dash='solid')  # Set the line style to solid
    )

trace1 = create_scatter_trace(sulphur_dioxide_emissions_data, 'Sulphur Dioxide / Sulphur Oxide Emissions', 'purple')
trace2 = create_scatter_trace(nitrogen_oxide_emissions_data, 'Nitrogen Oxide Emissions', 'orange')
trace3 = create_scatter_trace(GHG_Scope_1_data, 'GHG Scope 1', 'green')

# Combine all traces
all_traces = [trace1, trace2, trace3]

# Layout of the plot
layout = go.Layout(
    title='Normalized Data Over Time',
    xaxis=dict(title='Year'),
    yaxis=dict(title='Normalized Values', range=[0, 1]),
    legend=dict(yanchor="middle", y=0.5, xanchor="left", x=1.05)
)

# Create the figure with all traces and specify figure size
fig_Maersk_IN = go.Figure(data=all_traces, layout=layout)
fig_Maersk_IN.update_layout(width=1100, height=500)

# Define the colors for the regulations
regulation_colors = {"NFRD (2014/95/EU)": "RoyalBlue", "The Taxonomy Regulation": "Crimson"}

# Add dashed lines for the specific years (2018 and 2020) and corresponding dummy traces for the legend
for year, name, color in [("FY 2018", "NFRD (2014/95/EU)", regulation_colors["NFRD (2014/95/EU)"]),
                          ("FY 2020", "The Taxonomy Regulation", regulation_colors["The Taxonomy Regulation"])]:
    fig_Maersk_IN.add_shape(
        type="line",
        x0=year,
        y0=0,
        x1=year,
        y1=1,
        line=dict(color=color, width=2, dash="dashdot"),
    )
    fig_Maersk_IN.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        legendgroup=name,
        showlegend=True,
        name=name
    ))

# After all elements have been added, then you show the figure
fig_Maersk_IN.show()


# In[10]:


# Load the data

recap = pd.read_csv("/Users/pikpes/Maersk_PoS_Cleaned_Dataset_Taufik.csv")

# Selecting the required columns
columns = ['FY', 'ESG Disclosure Score', 'Environmental Disclosure Score', 'Governance Disclosure Score', 'Social Disclosure Score', 'GHG Scope 1', 'Loaded Volumes', 'Revenue', 'Nitrogen Oxide Emissions', 'Sulphur Dioxide / Sulphur Oxide Emissions']
new_df = recap[columns].copy()

# Defining a function to clean the numbers
def clean_number(x):
    if isinstance(x, str):
        return x.replace('.', '').replace(',', '.')
    return x

# Applying the function to the numerical columns
numerical_columns = ['ESG Disclosure Score', 'Environmental Disclosure Score', 'Governance Disclosure Score', 'Social Disclosure Score', 'GHG Scope 1', 'Loaded Volumes', 'Revenue', 'Nitrogen Oxide Emissions', 'Sulphur Dioxide / Sulphur Oxide Emissions']
for col in numerical_columns:
    new_df[col] = new_df[col].apply(clean_number).astype(float)
    
# Adding dummy variables for policy implementation and lags
new_df['Policy_2018'] = (new_df['FY'] >= 2018).astype(int)
new_df['Policy_2020'] = (new_df['FY'] >= 2020).astype(int)

models_in = []
dependent_variables = ['GHG Scope 1', 'Nitrogen Oxide Emissions', 'Sulphur Dioxide / Sulphur Oxide Emissions']
for dv in dependent_variables:
    y = new_df[dv]
    X = sm.add_constant(new_df[['Policy_2018', 'Policy_2020', 'Loaded Volumes', 'Revenue']])
    model = sm.OLS(y, X, missing='drop').fit()
    models_in.append(model)

# Creating the Stargazer object
stargazer = Stargazer(models_in)

# Customizing the Stargazer object
stargazer.title('Regression Results')
stargazer.custom_columns(['Model 1', 'Model 2', 'Model 3'], [1, 1, 1])
stargazer.show_model_numbers(False)
stargazer.add_custom_notes(['Model 1: GHG Scope 1', 'Model 2: Nitrogen Oxide Emissions', 'Model 3: Sulphur Dioxide / Sulphur Oxide Emissions'])

# Rendering the table in a Jupyter Notebook
stargazer


# In[ ]:





# ## External Indicators

# In[22]:


# Select the required columns
columns = ['FY', 'BESG ESG Score', 'BESG Environmental Pillar Score', 'BESG Social Pillar Score', 'BESG Governance Pillar Score', 'GHG Scope 1', 'Loaded Volumes', 'Revenue', 'Nitrogen Oxide Emissions', 'Sulphur Dioxide / Sulphur Oxide Emissions']
external_maersk = recap[columns].copy()

# Replace '-' with NaN
external_maersk.replace('—', pd.NA, inplace=True)

# Drop rows with any NaN values
external_maersk.dropna(inplace=True)

# Defining a function to clean the numbers
def clean_number(x):
    if isinstance(x, str):
        return x.replace('.', '').replace(',', '.')
    return x

# Apply the function to the numerical columns
for col in columns:
    external_maersk[col] = external_maersk[col].apply(clean_number).astype(float)


# In[39]:


variables_to_normalize = ['BESG ESG Score', 'BESG Environmental Pillar Score', 'BESG Social Pillar Score', 'BESG Governance Pillar Score',]

# Normalizing the variables
scaler = StandardScaler()
external_maersk[variables_to_normalize] = scaler.fit_transform(external_maersk[variables_to_normalize])

# Plotting
plt.figure(figsize=(12, 6))
for var in variables_to_normalize:
    plt.plot(external_maersk['FY'], external_maersk[var], marker='o', label=var)

plt.xlabel('Financial Year')
plt.ylabel('Normalized Values')
plt.title('Normalized BESG Trends Over Time')

# Placing the legend outside the plot on the right
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# # Save the plot as a PNG image
# plt.savefig('normalized_trends.png', bbox_inches='tight', dpi=300)

plt.show()


# In[41]:


# Add dummy variables for policy years
external_maersk['Policy_2018'] = (external_maersk['FY'] >= 2018).astype(int)
external_maersk['Policy_2020'] = (external_maersk['FY'] >= 2020).astype(int)

# Regression Analysis

models_external = []
dependent_variables = ['BESG ESG Score', 'BESG Environmental Pillar Score', 'BESG Social Pillar Score', 'BESG Governance Pillar Score']
for dv in dependent_variables:
    y = external_maersk[dv]
    X = sm.add_constant(external_maersk[['Policy_2018', 'Policy_2020', 'Loaded Volumes', 'Revenue']])
    model = sm.OLS(y, X, missing='drop').fit()
    models_external.append(model)

# Create and customize Stargazer object for displaying results
stargazer = Stargazer(models_external)
stargazer.title('Regression Results')
stargazer.custom_columns(['Model 1', 'Model 2', 'Model 3', 'Model 4'], [1, 1, 1, 1])
stargazer.show_model_numbers(False)
stargazer.add_custom_notes(['Model 1: BESG ESG Score', 'Model 2: BESG Environmental Pillar Score', 'Model 3: BESG Social Pillar Score', 'Model 4: BESG Governance Pillar Score'])

# Display the results table
stargazer


# In[ ]:




