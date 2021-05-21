#!/usr/bin/env python
# coding: utf-8

# ### NetFlix Dataset Exploration
# Tasks to perform:
# 1. Analysis which year has more type("TV Show or Movies").
# 2. Number of movies produced by directors.
# 3. Content available in different countries.
# 4. Countries with max and min Movie and TV Show

# ### Information of columns:
# 1. show_id:  Unique Id
# 2. type: Identifier - A Movie or TV Show
# 3. title: Title of the Movie / Tv Show
# 4. director: Director of the Movie
# 5. cast: Actors involved in the movie / show
# 6. country: Country where the movie / show was produced
# 7. date_added: Date it was added on Netflix
# 8. release_year: Actual Release year of the move / show
# 9. rating: TV Rating of the movie / show
# 10. duration: Total Duration - in minutes or number of seasons
# 11. listed_in: Genere
# 12. description: Description
# 

# ### Import necessary libraries

# In[179]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Analysis

# In[180]:


dataset = pd.read_csv("netflix_titles.csv")
dataset.head()


# In[254]:


dataset.info()  # Check datatypes of columns


# In[255]:


dataset.shape # Check number of rows and columns


# #### Total number of record: 7787

# In[183]:


dataset.isnull().sum()


# #### Columns with null values:
# 1. director: 31%
# 2. cast: 9%
# 3. country: 6.5%
# 4. rating: 0.001%

# # Task1: Analysis which year has more type("TV Show or Movies").

# In[184]:


task1_dataset = dataset.copy()      # Take copy of dataset, so our original will remain unchanged


# #### Column required for analysis of task1: Type and release_year

# #### Step 1: Check whether type and release_year have any null values

# In[256]:


task1_dataset.isna().sum()


# #### Step 2: Analyse the data

# In[186]:


task1_dataset[['type', 'release_year']]


# In[187]:


sns.countplot(x='type', data=task1_dataset)  ## Show the total number of movies and TV shows|


# In[190]:


sns.boxplot(x='release_year', y='type', orient="h", data=task1_dataset)


# ### Conclusion for Task 1: 
# 1. The data for both TV Shows and Movies are left skewed.
# 2. The movies started increasing since year 2003 whereas TV Shows start increasing after 2010.
# 3. 50% of the TV shows are after 2017 wheras 50 percent of the movies are after 2016

# # Task 2: Number of movies produced by directors

# In[191]:


task2_dataset= dataset.copy()


# In[192]:


task2_dataset.head()


# ### Step 1. Here we will consider columns: type, director, and release_year, so check whether they have null values

# In[193]:


task2_dataset.isna().sum()


# ### Step 2: As from the required column director column is having null values, handle missing value in that

# In[194]:


task2_dataset[task2_dataset['director'].isna()]


# In[195]:


### Here we know there is no way to fill the NaN value, so let us fill with 'Missing' value
task2_dataset['director'].fillna('Missing', inplace=True)


# In[196]:


task2_dataset.isna().sum()      #Now we can see all the directors are filled with 'Missing Values'


# In[197]:


task2_dataset[task2_dataset['type'] == 'TV Show']


# In[198]:


len(task2_dataset['director'].unique())


# ### will do analysis of first five directors, as there are lots of unique directors near by 4000+

# In[199]:


i = 0
for director_name in task2_dataset['director'].unique():
    if i == 5:
        break
    data_director = task2_dataset[task2_dataset['director'] == director_name]
    sns.countplot('director',hue='type', data=data_director)
    plt.show()
    i = i + 1
    


# # Task3: Content available in different countries.

# In[200]:


task3_dataset = dataset.copy()


# In[201]:


task3_dataset.head()


# In[202]:


## Columns required for contents in different countries: type, country, release_year


# In[203]:


task3_dataset.isna().sum()


# In[204]:


## Out of required columns type, country, release_year, country is having 507 null entries
task3_dataset.shape


# In[205]:


task3_dataset[task3_dataset['country'].isna()]


# In[206]:


### Let us drop the rows with country as null as those entries are vary less as percentage of null is 7%


# In[207]:


task3_dataset.dropna(subset=['country'], axis=0, inplace=True)


# In[208]:


task3_dataset.isna().sum()  # Now we have removed the null countries values.


# In[209]:


task3_dataset.head()


# In[210]:


relevant_data = task3_dataset[['type', 'country', 'release_year']]


# In[211]:


relevant_data.reset_index(inplace=True)
relevant_data.head(20)


# In[212]:


len(np.unique(relevant_data['country']))


# In[213]:


np.unique(relevant_data['country']) # Now here is a complication like in country we dont have single country, it also have multiple countries


# In[214]:


new_rows = []
for i in range(len(relevant_data)) :
    try:
        
        country_values = relevant_data.loc[i, "country"].split(',')
        if len(country_values) > 1:
            first_tym = True
            for value in country_values:
                if first_tym:
                    relevant_data.loc[i, "country"] = value
                    first_tym = False;
                else:
                    new = relevant_data.iloc[i]
                    new['country'] = value
                    new_rows.append(new)
    except:
        print('value at error is : ', i)
        
relevant_data.append(new_rows)


# In[215]:


# As there are lot of countries, so just looked for few over here
i=0;
for country in np.unique(relevant_data['country']):
    if i == 5:
        break;
    i=i+1
    print(country)
    sns.countplot(x= 'type', data=relevant_data[relevant_data['country'] == country])
    plt.title(country)
    plt.show()
    
    


# ### Task 4: Countries with Maximum and Minimum movies and Tv shows

# In[216]:


task4_dataset = dataset.copy()


# In[217]:


task4_dataset


# #### Countries with maximum TV Shows and movies. Columns required are country and type

# In[234]:


country_type_data = task4_dataset[['type', 'country']]


# In[235]:


country_type_data.isna().sum() # Here we see country is having null values


# In[236]:


# As we can see we just have only 6.5 percent values of country as null, so we can drop those rows
percentage_country_null = (country_type_data[country_type_data['country'].isna()].shape[0]/country_type_data.shape[0]) * 100
percentage_country_null


# In[237]:


country_type_data.dropna(axis=0, subset=['country'], inplace=True)


# In[238]:


country_type_data.isna().sum() # we don't have any null values


# In[239]:


country_type_data.reset_index(inplace=True) # As we removed rows, so we need to reset index|


# In[240]:


country_type_data.head()


# #### As previous tasks, here also country columns are having comma, so lets make them as new rows for comma seperated countries

# In[241]:


new_rows = []
for i in range(len(country_type_data)) :
    try:
        
        country_values = country_type_data.loc[i, "country"].split(',')
        if len(country_values) > 1:
            first_tym = True
            for value in country_values:
                if first_tym:
                    country_type_data.loc[i, "country"] = value
                    first_tym = False;
                else:
                    new = country_type_data.iloc[i]
                    new['country'] = value
                    new_rows.append(new)
    except:
        print('value at error is : ', i)
        
country_type_data.append(new_rows)


# In[242]:


encoded_data = pd.get_dummies(country_type_data['type'])


# In[243]:


country_type_data


# In[244]:


country_type_data['Movie']=encoded_data['Movie']
country_type_data['TV Show']=encoded_data['TV Show']


# In[245]:


country_type_data.drop('type', axis=1, inplace=True)


# In[246]:


country_type_data


# In[247]:


grouped_data = country_type_data.groupby('country').sum()


# ### United States has maximum number of movies

# In[249]:


grouped_data[grouped_data['Movie'] == grouped_data['Movie'].max()]


# #### There are 7 countries with no movies

# In[251]:


grouped_data[grouped_data['Movie'] == grouped_data['Movie'].min()]


# #### There are lot of countries with no "TV Show"

# In[252]:


grouped_data[grouped_data['TV Show'] == grouped_data['TV Show'].min()]


# #### United States is the country with maximum number of "TV Show"

# In[253]:


grouped_data[grouped_data['TV Show'] == grouped_data['TV Show'].max()]


# In[ ]:




