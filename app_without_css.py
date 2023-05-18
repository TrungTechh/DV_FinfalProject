import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import pandas as pd
import numpy as np
from scipy import signal

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline



# global variable
df = pd.read_csv('Life_Expectancy_Data.csv')
years = pd.unique(df.Year)

def read_data():
    # I. Read data
    st.markdown('# Read the data')

    # 1.About data
    st.markdown('## About the data')
    aboutdata = '''
    The Global Health Observatory (GHO) data repository under World Health Organization (WHO) keeps track of 
    the health status as well as many other related factors for all countries The datasets are made available 
    to public for the purpose of health data analysis. The dataset related to life expectancy, health factors 
    for 193 countries has been collected from the same WHO data repository website and its corresponding economic 
    data was collected from United Nation website. Among all categories of health-related factors only those critical 
    factors were chosen which are more representative. It has been observed that in the past 15 years , there has been 
    a huge development in health sector resulting in improvement of human mortality rates especially in the developing 
    nations in comparison to the past 30 years.
    '''
    st.markdown(aboutdata)

    data_description = '''
    #### Feature information

    | Heeader name                    | Description |
    | ------------------------------- | ----------- |
    | Country                         |             |
    | Year                            |             |
    | Status                          |             |
    | Life expectancy                 |             |
    | Adult Mortality                 |             |
    | infant deaths                   |             |
    | Alcohol                         |             |
    | percentage expenditure          |             |
    | Hepatitis B                     |             |
    | Measles                         |             |
    | BMI                             |             |
    | under-five deaths               |             |
    | Polio                           |             |
    | Total expenditure               |             |
    | Diphtheria                      |             |
    | HIV/AIDS                        |             |
    | GDP                             |             |
    | Population                      |             |
    | thinness 1-19 years             |             |
    | thinness 5-9 years              |             |
    | Income composition of resources |             |
    | Schooling                       |             |
    '''
    st.markdown(data_description)

def descriptive_statistic():
    st.markdown('# Descriptive Statistic')

    num_stat = df.describe().T
    st.markdown('## Numerical statistics')
    st.dataframe(round(num_stat, 2), width=2000, height=735)


    cat_stat = df.describe(include=['O'])
    st.markdown('## Categorical statistics')
    st.dataframe(cat_stat)

#--------------------------data exploration--------------------------
#-------------univariate analysis-------------
def univariate_lifeExpectancy():
    st.markdown('### Life expectancy')
    fig, axes = plt.subplots(4, 4, figsize=(25, 20))

    for i in range(len(years)):
        LE_year = df.loc[df.Year == years[i], ['Country', 'Life expectancy']]
        ax = axes[i // 4][i - 4*(i // 4)]
        ax.hist(LE_year['Life expectancy'], bins=15)
        ax.axvline(LE_year['Life expectancy'].mean(), c='r', linestyle='--', label='average')
        ax.axvline(LE_year.loc[df.Country == 'Viet Nam', 'Life expectancy'].values[0], c='y', linestyle='-.', label='Vietnam')
        ax.set_title(f'Year {years[i]}')

    fig.suptitle('Distribution of life expectancy in 2000-2015', horizontalalignment='center', verticalalignment='center', fontsize = 25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize='xx-large')

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.2)

    st.pyplot(fig)

    comment = '''
    **Comment:**
    - It may not follow the normal distribution.
    - Life expectation (LE) in the world is from 40 to over 85 with the higher rate is in 70-75 years old.
    - Most of life expectation in the world is higher than mean.
    - LE of Viet Nam is about 73 and always higher than average LE.
    - In the late of 2000-2015 duration, the LE increase lower bound to 50.
    - If thinking normally, the LE will increase by the time because of development of science and technology. \
        However, if early year histograms are left skew (means that the long life rate is higher than short life) \
        then the later year histograms are not skewed (means that the long life rate is decreased)
    '''
    st.markdown(comment)

def univariate_status():
    st.markdown('### Developed or Developing')

    ded_ding = pd.pivot_table(df, index='Status', columns='Year', aggfunc='size')
    st.write(ded_ding)

    fig = plt.figure()
    plt.pie(ded_ding[2000], labels=ded_ding.index.tolist(), explode=[0, 0.1], autopct='%0.0f%%')
    plt.title("Percent developing vs developed countries")
    st.pyplot(fig)

    comment = '''
    **Comment:**
    - There are 32 developed countries and 151 developing one.
    - Moreover, there is no country devoped to became a developed country in 2000-2015 duration
    '''
    st.markdown(comment)

def univariate_population():
    st.markdown('### Population')
    fig, axes = plt.subplots(4, 4, figsize=(25, 20))

    for i in range(len(years)):
        population_year = df.loc[df.Year == years[i], ['Country', 'Population']]
        ax = axes[i // 4][i - 4*(i // 4)]
        ax.hist(population_year['Population'], bins=15)
        ax.axvline(population_year['Population'].mean(), c='r', linestyle='--', label='average')
        ax.axvline(population_year.loc[df.Country == 'Viet Nam', 'Population'].values[0], c='y', linestyle='-.', label='Vietnam')
        ax.set_title(f'Year {years[i]}')

    fig.suptitle('Distribution of Population in 2000-2015', horizontalalignment='center', verticalalignment='center', fontsize = 25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize='xx-large')

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.2)
    st.pyplot(fig)

    st.write('Countries with large populations during this period')
    average_population = df.groupby('Country', as_index=False)['Population'].mean()
    st.write(average_population.loc[average_population.Population > (0.5 * 1e8)])

    comment = '''
    **Comment:**
    - Most countries in the world have a population of about 10 million people in 2000-2015 duration.
    - Viet Nam is also like that.
    - There are some outliers such as Brazil, India, Indonesia, Nigeria, Pakistan, and Russian Federation	
    '''
    st.markdown(comment)

def univariate_gdp():
    st.write('### GDP')
    fig, axes = plt.subplots(4, 4, figsize=(25, 20))

    for i in range(len(years)):
        gpd_year = df.loc[df.Year == years[i], ['Country', 'GDP']]
        ax = axes[i // 4][i - 4*(i // 4)]
        ax.hist(gpd_year['GDP'], bins=15)
        ax.axvline(gpd_year['GDP'].mean(), c='r', linestyle='--', label='average')
        ax.axvline(gpd_year.loc[df.Country == 'Viet Nam', 'GDP'].values[0], c='y', linestyle='-.', label='Vietnam')
        ax.set_title(f'Year {years[i]}')

    fig.suptitle('Distribution of GDP in 2000-2015', horizontalalignment='center', verticalalignment='center', fontsize = 25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize='xx-large')

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.2)
    st.pyplot(fig)


    average_gdp = df.groupby('Country', as_index=False)['GDP'].mean().sort_values(by='GDP')
    pd.options.display.float_format = '{:,.2f}'.format
    average_gdp['type'] = average_gdp.GDP >= 10000
    type_gdp = average_gdp.groupby('type').agg({'GDP':['count', 'sum']})

    fig, [ax0, ax1] = plt.subplots(1, 2)
    ax0.pie(type_gdp[('GDP', 'count')], labels=['Low GDP', 'High GDP'], explode=[0, 0.1], autopct='%0.0f%%')
    ax0.set_title('Number of countries')

    ax1.pie(type_gdp[('GDP',   'sum')], labels=['Low GDP', 'High GDP'], explode=[0, 0.1], autopct='%0.0f%%')
    ax1.set_title('Percent GDP')

    plt.suptitle('Compare the number of countries with low GDP and high GDP\n and the proportions between them',\
                 horizontalalignment='center', verticalalignment='center', fontsize = 25)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.2)
    st.pyplot(fig)

    comment = '''
    **Comment:**
    - Most of countries in the world have too low GPD in this duration, are under 10000.
    - Viet Nam has equal or higher than world average GDP but still be low.
    - From second plot in this part, only 17% of countries in the world have high GDP but account for 59% of the world's total GDP.
    '''
    st.markdown(comment)

def univariate_bmi():
    st.markdown('### BMI')
    fig, axes = plt.subplots(4, 4, figsize=(25, 20))

    for i in range(len(years)):
        bmi_year = df.loc[df.Year == years[i], ['Country', 'BMI']]
        ax = axes[i // 4][i - 4*(i // 4)]
        ax.hist(bmi_year['BMI'], bins=15)
        ax.axvline(bmi_year['BMI'].mean(), c='r', linestyle='--', label='average')
        ax.axvline(bmi_year.loc[df.Country == 'Viet Nam', 'BMI'].values[0], c='y', linestyle='-.', label='Vietnam')
        ax.set_title(f'Year {years[i]}')

    fig.suptitle('Distribution of GDP in 2000-2015', horizontalalignment='center', verticalalignment='center', fontsize = 25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize='xx-large')

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.2)
    st.pyplot(fig)

    fig = plt.figure()
    average_bmi = df.groupby('Country', as_index=False)['BMI'].mean()
    average_bmi['type'] = average_bmi['BMI'].copy()

    average_bmi.loc[average_bmi.BMI < 18.5, 'type'] = 'thin'
    average_bmi.loc[(average_bmi.BMI >= 18.5) & (average_bmi.BMI < 25), 'type'] = 'normal'
    average_bmi.loc[average_bmi.BMI >= 25, 'type'] = 'fat'

    type_bmi = average_bmi.type.value_counts()
    plt.pie(type_bmi, labels=type_bmi.index.tolist(), explode=[0.1, 0, 0], autopct='%0.0f%%')
    plt.title('BMI type of world population')
    st.pyplot(fig)

    comment = '''
    **Comment:**
    - World BMI is from over 10 to over 70
    - There are a lot of outliers that are smaller than 10, which means that there are a lots countries have too bad BMI indicators
    - The distribution of MBI look like combination of 2 normal distributions, one is around 20-30 and the other is around 50-60. \
    This mean that there is a large distance BMI between 2 groups of the world
    - Another useful information that most of countries is fat according their BMI, there are 71% countries of the world is fat BMI
    ***&rarr; Fat BMI could be one of the strongest reason affect to life expectancy directly***
    '''
    st.markdown(comment)

def univariate_analysis():
    st.markdown('## Univariate analysis')
    univariate_lifeExpectancy()
    univariate_status()
    univariate_population()
    univariate_gdp()
    univariate_bmi()

#-------------multivariate analysis-------------
def multivariate_analysis():
    st.markdown('## Multivariate analysis')

#-------------time-series analysis--------------
def timeseries_analysis():
    st.markdown('## Time - series analysis')
    st.markdown('### Are life expectancy stationary?')

    df = pd.read_csv('LE_cleaned_data.csv')
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')

    time = df.pivot(index='Year', columns='Country', values='Life expectancy')
    vn = time['Viet Nam']
    vn.plot(kind='line', figsize=(10, 5));
    st.line_chart(vn)

    test = vn.reset_index()
    df_station = adfuller(test['Viet Nam'], autolag='AIC')
    st.markdown('Hypothesis Testing by using p-value. \
                If p-value is less than $alpha = 0.05$, we can reject the null hypothesis \
                that the time series is non-stationary.')
    st.text('P-value: ' + str(df_station[1]))
    st.markdown('We calculated p-value = 0.5 > 0.05, \
                then we reject H1 and conclude that the time series is non-stationary (accept H0).')
    time = time.T
    world = time.describe().round(2)
    world = world.loc[world.index == 'mean']
    world = world.T.rename({'mean': 'World'}, axis=1)
    merged = world.reset_index().merge(vn.reset_index(),left_on = 'Year', right_on = 'Year', how = 'inner')
    merged.set_index('Year', inplace=True)
    # merged.plot(kind='line', figsize=(20, 10), title='World and Viet Nam Life Expectancy');
    st.line_chart(merged)

def data_exploration():
    st.markdown('# Data exploration')
    univariate_md = '''
    ## Univariate analysis
    In this part, we just explore common factors which may affect directly to life expectancy such as:
    - `Life expectancy`: of course, how can we not analysis this variable
    - `Status`: Easily, we can think that developed countris's life expectancy may be higher than developing one
    - `Population`: longer life, denser population
    - `GDP`: higher GDP, goverment could spend more money for medical and people have more money for their health
    - `BMI`: a basic indicator for country health.
    '''
    st.markdown(univariate_md)

    univariate_analysis()

#--------------------------regression analysis--------------------------
# predict life expectancy
train, test = df.loc[df.Year < 2014], df.loc[df.Year >= 2014]
X_train, y_train = train.drop('Life expectancy', axis=1), train.loc[:, ['Life expectancy']]
X_test, y_test = test.drop('Life expectancy', axis=1), test.loc[:, ['Life expectancy']]

def train_model():
    cat_cols = ['Country', 'Status']
    num_cols = ['Year'] + X_train.columns.tolist()[3:]
    cat_pipe = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
                            OneHotEncoder(handle_unknown='ignore'))
    num_pipe = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'))
    fill_missing = make_column_transformer(
        (num_pipe, num_cols),
        (cat_pipe, cat_cols)
    )
    preprocess_pipeline = Pipeline([
        ('fill_missing', fill_missing),
        ('scaler', StandardScaler(with_mean=False))
    ])

    lr = LinearRegression()

    pl = Pipeline([('preprocessing', preprocess_pipeline), ('regression', lr)])
    last_model = pl.fit(df.drop('Life expectancy', axis=1), df[['Life expectancy']])

    return last_model

def predict_lifeExpectancy():
    st.markdown('## Predict life expectancy in the future')
    country = [['Viet Nam'] * (2031 - 2016)]
    year = [list(range(2016, 2031, 1))]
    nan_arr = [np.full(2031 - 2016, np.nan).tolist() for i in range(19)]
    X = dict(zip(X_train.columns, country + year + nan_arr))
    X = pd.DataFrame(X)

    # predict
    last_model = train_model()
    predictions = last_model.predict(X)

    # visualization
    fig = plt.figure()
    plt.plot(list(range(2016, 2031)), predictions, marker='o', markerfacecolor='red')
    for x in range(2016, 2031, 2):
        plt.text(x, predictions[x - 2016] - 0.2, round(predictions[x - 2016][0], 2), ha='center', va='bottom')
    plt.xlabel('Year')
    plt.ylabel('Life expectancy prediction')
    plt.title('Prediction life expectancy of Viet Nam from 2016 - 2030')

    plt.pyplot(fig)


# What factors influence life expectancy the most
def most_influence():
    st.markdown('## What factors influence life expectancy the most?')
    feature_index = ["HIV/AIDS", "Adult Mortality", "Income composition of resources", "Schooling", "thinness 5-9 years"]
    feature_value = [0.379045, 0.280811, 0.209,0.030993, 0.016]


    fig = plt.figure(figsize=(10, 4))
    sns.barplot(x=feature_value, y=feature_index)
    plt.xlabel('Feature Importance Score')
    plt.title("Visualizing Important Features")
    st.pyplot(fig)


    st.markdown("If a country want to improve their life expectency, the most important thing they should are:")

    with st.container():
        st.markdown("- Prevent HIV/AIDS ")
        st.markdown("- Care in take care of adults")
        st.markdown("- Optimal utilization of available resources")
        st.markdown("- Invest in education")
        st.markdown("- Focus on nutrition for children 5 - 9 years")


# Does the higher the GDP, the longer the life expectancy of the country will increase?
def higherGDP_longerLE():
    st.markdown('## Does the higher the GDP, the longer the life expectancy of the country will increase?')

def regression_analysis():
    st.markdown('# Regression analysis')
    # predict_lifeExpectancy()
    most_influence()
    higherGDP_longerLE()


#--------------------------Solution--------------------------
# What factor should be changed to increase life expectancy?
def factor_change():
    st.markdown('## What factor should be changed to increase life expectancy?')

# For Vietnam in particular, what factors need to be changed most to increase the average life expectancy?
def factor_change_Vietnam():
    st.markdown('## For Vietnam in particular, what factors need to be changed most to increase the average life expectancy?')

def solution():
    st.markdown('# Solution')
    factor_change()
    factor_change_Vietnam()


#--------------------------run--------------------------
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

read_data()
descriptive_statistic()
data_exploration()
timeseries_analysis()
regression_analysis()
solution()