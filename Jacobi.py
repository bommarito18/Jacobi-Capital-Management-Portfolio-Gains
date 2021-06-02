#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime, timedelta
from matplotlib import dates as mpl_dates
import plotly.graph_objects as go


# In[2]:


data = pd.read_csv('../Data/data1.csv')
data


# In[3]:


Date = data['Date']
#ETF = data['ETF']


# In[4]:


In = data['In']
Out = data['Out']


# In[5]:


data['Profit'] = Out - In
data


# In[6]:


Profit = data['Profit']


# In[7]:


data['Acc'] = Profit.cumsum()
data


# In[8]:


Acc = data['Acc']


# In[9]:


data['diff'] = Acc.diff()
data


# In[10]:


diff = data['diff']


# In[11]:


data['pct'] = Profit / In * 100
data


# In[12]:


pct = data['pct']


# In[13]:


data['pctacc'] = pct.cumsum()
data


# In[14]:


pctacc = data['pctacc']


# In[15]:


data['p'] = pctacc / 100
p = data['p']
data


# In[17]:


Dollar_Value = int(input("Dollar Value: $"))


# In[18]:


data['DV'] = Dollar_Value * p + Dollar_Value
DV = data['DV']
data


# In[19]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters ()

fig, ax = plt.subplots(figsize=(18, 10))
ax.tick_params(axis='x', colors='navy')
ax.tick_params(axis='y', colors='green')

plt.annotate('Program by: Taylor Bommarito', xy=(0.035, 0.86), xycoords='axes fraction', color='navy', fontsize=12)
plt.annotate('Dollar Gained : ${:.2f}'.format(data.loc[data.index[-1], 'DV']), color = 'navy', fontsize=24, weight='bold', xy=(0.38, 1.08), xycoords='axes fraction')


plt.rc('axes',edgecolor='navy')
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
Date = data['Date']

plt.plot_date(Date, DV, color='green', label='Return in $', linestyle=(0,(5, 10)), markersize=0)
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.legend( loc='upper left', prop={"size":15})
plt.xlabel('Annual Date', color = 'navy', fontsize=20, weight='bold')
plt.ylabel('$ for Portfolio Returns (dashed lines)', color = 'green', fontsize=20, weight='bold')

ax1 = ax.twinx()
curve1 = plt.plot_date(Date, pctacc, label='Return in %', color='red', linestyle= 'solid', linewidth=4, markersize=10)
plt.ylabel('% for Portfolio Returns (solid line)', color = 'red', fontsize=20, weight='bold')
ax1.tick_params(axis='y', colors='red')
plt.legend(loc='lower right', prop={"size":15})

plt.title('Portfolio Gains : {:.2f}%'.format(data.loc[data.index[-1],'pctacc']), color = 'navy', fontsize=24, weight='bold', pad=100)
plt.show()


# In[20]:


from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces

fig.add_trace(
    go.Scatter(x=data['Date'], y=data['pctacc'], name="Percent Gained",line=dict(color="#0D2A63"), mode="lines+markers"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=data['Date'], y=data['DV'], name="Amount Gained In $", line=dict(color="rgb(255,237,111)"), mode="markers"),
    secondary_y=True,
)


# Add figure title
fig.update_layout(
   #title_text="Interactive Buy / Sell"
   title_text=('<b>Jacobi Capital Portfolio Gains : {:.2f}%</b>'.format(data.loc[data.index[-2],'pctacc']))
)


# Set x-axis title
fig.update_xaxes(title_text="<b>Annual Date</b>")

# Set y-axes titles
fig.update_yaxes(title_text="<b>%</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>$</b>", secondary_y=True)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.update_xaxes(rangeslider_visible=True)

fig.update_layout( width=1500, height=700)

fig.update_layout(plot_bgcolor='white')

fig.update_xaxes(showline=True, linewidth=2, linecolor='#0D2A63', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='#0D2A63', mirror=True)

fig.show()


# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])


app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter


# In[ ]:





# In[ ]:




