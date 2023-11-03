#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load dataset

athletes = pd.read_csv('C:/Users/ROHIT SONAWANE/Desktop/untitled folder 1/athlete_events.csv')
region = pd.read_csv('C:/Users/ROHIT SONAWANE/Desktop/untitled folder 1/noc_regions.csv')


# In[3]:


athletes.head()


# In[4]:


region.head()


# In[5]:


#join the dataframes 

athletes_df = athletes.merge(region, how = 'left', on = 'NOC')
athletes_df.head()


# In[6]:


athletes_df.shape


# In[7]:


# column name consistent 
 
athletes_df.rename(columns={'region':'Region' , 'notes':'Notes'}, inplace=True);


# In[8]:


athletes_df.head()


# In[9]:


athletes_df.info()


# In[10]:


athletes_df.describe()


# In[11]:


# check null values
nan_values = athletes_df.isna()
nan_columns = nan_values.any()
nan_columns


# In[12]:


athletes_df.isnull().sum()


# In[13]:


# India details

athletes_df.query('Team == "India"').head(5)


# In[14]:


#Japan details

athletes_df.query('Team == "Japan"').head(5)


# In[15]:


#top 10 country participating

top_10_countries = athletes_df.Team.value_counts().sort_values(ascending=False).head(10)
top_10_countries


# In[16]:


top_10_countries = pd.DataFrame({'Country': ['Country India', 'Country Canada', 'Country Japan', 'Country united states', 'Country Jermany', 'Country Frans', 'Country Africa', 'Country Australia', 'Country Hungary', 'Country Italy' ],
                                'Participation': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10 ]})

plt.figure(figsize=(12, 6))
plt.title('Overall Participation by Country')
sns.barplot(x='Country', y='Participation', data=top_10_countries, palette='Set1')
plt.xticks(rotation=2500)
plt.show()


# In[17]:


# Age Distribution of the aprticipats

plt.figure(figsize=(12, 6))
plt.title("Age distribution of the athletes")
plt.xlabel('Age')
plt.ylabel('Number of participants')
plt.hist(athletes_df['Age'], bins=np.arange(10, 80, 2), color='orange', edgecolor='white')
plt.show()


# In[18]:


winter_sports = athletes_df[athletes_df['Season'] == 'Winter']['Sport'].unique()
winter_sports


# In[19]:


summer_sports = athletes_df[athletes_df['Season'] == 'summer']['Sport'].unique()
summer_sports


# In[21]:


gender_counts = [50, 70]  
labels = ['Male', 'Female']


# In[22]:


# pie plot for male and female paticipate

plt.figure(figsize=(12, 6))
plt.title('Gender Distribution')
plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=150, shadow=True)
plt.show()


# In[23]:


#Total medals

athletes_df.Medal.value_counts()


# In[26]:


female_participates = athletes_df[(athletes_df['Sex'] == 'F') & (athletes_df['Season'] == 'Summer')][['Sex', 'Year']]
female_participate = female_participates.groupby('Year').count().reset_index()
female_participate.tail()


# In[28]:


womenolympics = athletes_df[(athletes_df['Sex'] == 'F') & (athletes_df['Season'] == 'Summer')]


# In[30]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=womenolympics, palette="Spectral")
plt.title('Women Participation')
plt.show()


# In[32]:


part = womenolympics.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(20, 10))
part.loc[:, 'F'].plot()
plt.title('Plot of Female Athlete over Time')
plt.xlabel('Year')
plt.ylabel('Number of Female Athletes')
plt.show()


# In[46]:


# gold medal athlete

goldMedals = athletes_df[(athletes_df.Medal == 'Gold')]
goldMedals.head()


# In[48]:


max_year = athletes_df.Year.max()
print(max_year)

team_names = athletes_df[(athletes_df.Year == max_year) & (athletes_df.Medal == 'Gold')].Team

team_names.value_counts().head(10)


# In[52]:


sns.barplot(x=team_names.value_counts().head(20), y=team_names.value_counts().head(20).index)

plt.ylabel(None);
plt.xlabel('Countrywise Medals for the year 2016');


# In[54]:


not_null_medals = athletes_df[(athletes_df['Height'].notnull()) & (athletes_df['Weight'].notnull())]


# In[55]:


plt.figure(figsize = (12,10))
axis = sns.scatterplot(x="Height", y="Weight", data=not_null_medals, hue="Sex")
plt.title('Height vs Weight of Olympic Medalists')


# In[ ]:




