import streamlit as st
import streamlit.components.v1 as components
import mpld3
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

    | Heeader name                    | Description                                                              |
    | ------------------------------- | -------------------------------------------------------------------------|
    | Country                         | 193 countries in around the world                                        |
    | Year                            | From 2000 to 2015                                                        |
    | Status                          | Developed or Developing status                                           |
    | Life expectancy                 | Life Expectancy in age                                                   |
    | Adult Mortality                 | Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)                                     |
    | infant deaths                   | Number of Infant Deaths per 1000 population                              |
    | Alcohol                         | Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)                           |
    | percentage expenditure          | Expenditure on health of Gross Domestic Product per capita(%)            |
    | Hepatitis B                     | Hepatitis B (HepB) immunization coverage among 1-year-olds(%)           |
    | Measles                         | Measles - number of reported cases per 1000 population                   |
    | BMI                             | Average Body Mass Index of entire population                             |
    | under-five deaths               | Number of under-five deaths per 1000 population                          |
    | Polio                           | Polio (Pol3) immunization coverage among 1-year-olds (%)                 |
    | Total expenditure               | General government expenditure on health as total (%)                    |
    | Diphtheria                      | Diphtheria tetanus toxoid and pertussis immunization coverage among 1-year-olds (%)        |
    | HIV/AIDS                        | Deaths per 1 000 live births HIV/AIDS (0-4 years)                        |
    | GDP                             | Gross Domestic Product per capita (in USD)                               |
    | Population                      | Population of the country                                                |
    | thinness 1-19 years             | Prevalence of thinness among children and adolescents 10 to 19 Age (%)   |
    | thinness 5-9 years              | Prevalence of thinness among children for Age 5 to 9(%)                  |
    | Income composition of resources | Human Development Index of income composition of resources (0 to 1)              |
    | Schooling                       | Number of years of Schooling(years)                                      |

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
def plot1(df_life, features, title='Features', columns=2, x_lim=None):
    rows = math.ceil(len(features) / 2)
    fig, ax = plt.subplots(rows, columns, sharey=True)
    for i, feature in enumerate(features):
        ax = plt.subplot(rows, columns, i + 1)
        sns.regplot(data=df_life, x=feature, y='Life expectancy', scatter_kws={'s': 60, 'edgecolor': 'k'},
                    line_kws={'color': 'red'}, ax=ax)
        ax.set_title('Life Expectancy vs ' + feature)
        
    fig.suptitle('{} x Life Expectancy'.format(title), fontsize=25, x=0.56)
    fig.tight_layout(rect=[0.05, 0.03, 1, 1])
    st.pyplot(fig)

# Function to plot scatter plots with regression lines but usse log scale 
def plot2(df_life, features, title='Features', columns=2, x_lim=None):
    rows = math.ceil(len(features) / 2)
    fig, ax = plt.subplots(rows, columns, sharey=True)
    
    for i, feature in enumerate(features):
        ax = plt.subplot(rows, columns, i + 1)
        log_feature = np.log1p(df_life[feature])
        sns.regplot(data=df_life, x=log_feature, y='Life expectancy', scatter_kws={'s': 60, 'edgecolor': 'k'},
                    line_kws={'color': 'red'}, ax=ax)
        ax.set_title('Life Expectancy vs log(' + feature + ')')
        
    fig.suptitle('{} x Life Expectancy'.format(title), fontsize=25, x=0.56)
    fig.tight_layout(rect=[0.05, 0.03, 1, 1])
    st.pyplot(fig)
#Function to plot scatter plots
def plot_scatterplot(df_life, features, title = 'Features', columns = 2, x_lim=None):
    
    rows = math.ceil(len(features)/2)

    fig, ax = plt.subplots(rows, columns, sharey = True)
    
    for i, feature in enumerate(features):
        ax = plt.subplot(rows, columns, i+1)
        sns.scatterplot(data = df_life,
                        x = feature,
                        y = 'Life expectancy',
                        hue = 'Status',
                        palette=['#669bbc', '#c1121f'],
                        ax = ax)
        if (i == 0):
            ax.legend()
        else:
            ax.legend("")
        
    fig.legend(*ax.get_legend_handles_labels(), 
               loc='lower center', 
               bbox_to_anchor=(1.04, 0.5),
               fontsize='small')
    fig.suptitle('{} x Life Expectancy'.format(title), 
                 fontsize = 25, 
                 x = 0.56);

    fig.tight_layout(rect=[0.05, 0.03, 1, 1])
    st.pyplot(fig)
def multivariate_analysis():
    st.markdown('## Multivariate analysis')
    df_life = df.copy()
    columns_name_fixed = []

    for column in df.columns:
        if column == ' thinness  1-19 years':
            column = 'Thinness 1-19 years'
        else:
            column = column.strip(' ').replace("  ", " ")
            column = column[:1].upper() + column[1:]
        columns_name_fixed.append(column)
    df_life.columns = columns_name_fixed

    st.markdown('### The relationships of the life expectancy with the other independent variables')
    
    # List of positively correlated features with life expectancy
    pos_correlated_features = ['Income composition of resources', 'Schooling', 'GDP', 'Total expenditure', 
                            'BMI', 'Diphtheria']
    # Plot scatter plots with regression lines for positively correlated features
    plot1(df_life, pos_correlated_features, title='Positively Correlated Features')
    # List of negatively correlated features with life expectancy
    neg_correlated_features = ['Adult Mortality', 'HIV/AIDS', 
                                'Thinness 1-19 years', 'Infant deaths']
    # Plot scatter plots with regression lines for nagatively correlated features
    plot1(df_life, neg_correlated_features, title='Negatively Correlated Features')
    #Check other correlations
    features = ['Population', 'Alcohol']
    plot1(df_life, features)

    # List of positively correlated features with life expectancy
    positively_correlated_features = ['Income composition of resources', 'Schooling', 'GDP', 'Total expenditure', 
                            'BMI', 'Diphtheria']
    # Plot scatter plots with regression lines for the logarithm of positively correlated features
    plot2(df_life, positively_correlated_features, title='Positively Correlated Features (log scale)')
    # List of negatively correlated features with life expectancy
    neg_correlated_features = ['Adult Mortality', 'HIV/AIDS', 
                                'Thinness 1-19 years', 'Infant deaths']
    # Plot scatter plots with regression lines for the logarithm of negatively correlated features
    plot2(df_life, neg_correlated_features, title='Negatively Correlated Features (log scale)')
    #Check other correlations
    features = ['Population', 'Alcohol']
    plot2(df_life, features, title='Other Features (log scale)')

    st.markdown('### The relationships of the life expectancy point with the other independent variables group by country status')

    #Plot Life Expectancy x positively correlated features
    pos_correlated_features = ['Income composition of resources', 'Schooling', 
                            'GDP', 'Total expenditure', 
                            'BMI', 'Diphtheria']
    title = 'Positively correlated features'
    plot_scatterplot(df_life, pos_correlated_features, title)
    #Plot Life Expectancy x negatively correlated features
    neg_correlated_features = ['Adult Mortality', 'HIV/AIDS', 
                            'Thinness 1-19 years', 'Infant deaths']
    title = 'Negatively correlated features'
    plot_scatterplot(df_life, neg_correlated_features, title)
    #Check other correlations
    df_temp = df_life.loc[df_life['Population'] <= 1*1e7, :] 
    features = ['Population', 'Alcohol']
    plot_scatterplot(df_temp, features)
    st.markdown('''
    #### Conclusions:
        - It seems that the absolute number of a country's population does not have a direct relationship with life expectancy. Perhaps a more interesting variable would be population density, which can provide more clues about the country's social and geographical conditions.
        - Another interesting point is that countries with the highest alcohol consumption also have the highest life expectancies. However, this seems to be the classic case for using the maxim 'Correlation does not imply causation'. The life expectancy of someone who owns a Ferrari is possibly higher than that of the rest of the population, but that does not mean that buying a Ferrari will increase their life expectancy. The same applies to alcohol. One hypothesis is that in developed countries, the population's average has better financial conditions, allowing for greater consumption of luxury goods such as alcohol.
        - After checking the relationships of the dependent variable with the independent variables, it is important to analyze the distribution of these variables. Through them, it is possible to have the first clues if there are outliers in the dataset.    
    ''')


#-------------time-series analysis--------------
def timeseries_analysis():
    st.markdown('## Time - series analysis')
    
    #-----------------Stationarity-----------------
    st.markdown('### Are life expectancy stationary?')
    df = pd.read_csv('LE_cleaned_data.csv')
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')

    time = df.pivot(index='Year', columns='Country', values='Life expectancy')
    vn = time['Viet Nam']
    sns.set_style('white')
    fig = plt.figure()
    plt.plot(vn, label='Viet Nam');
    plt.title('Life expectancy of Viet Nam')
    plt.xlabel('Year')
    plt.ylabel('Life expectancy')

    # st.pyplot(fig)
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)

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

    fig = plt.figure()
    plt.plot(merged.index, merged['Viet Nam'], label='Viet Nam');
    plt.plot(merged.index, merged['World'], label='World');
    plt.title('Life expectancy of Viet Nam and World')
    plt.xlabel('Year')
    plt.ylabel('Life expectancy')
    plt.legend()
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)

    st.markdown('''
        Conclusion:
            - Life Expectancy tends to increase over time.
            - This time-series data is non-stationary.
        ### Cyclical Analysis
        Is the time-series data cyclical? Let us explore first with autocorrelation by ACF and PACF visualization.''')
    
    #-----------------Cyclical-----------------
    fig, ax = plt.subplots()
    acf_cal = world.reset_index().rename(columns={'World':'Life Expectancy'})
    # mpl.rc("figure")
    plot_acf(acf_cal["Life Expectancy"], lags = 15, ax=ax);
    st.pyplot(fig)

    fig,ax = plt.subplots()
    plot_pacf(acf_cal["Life Expectancy"], lags = 7, method='ywm', ax=ax);
    st.pyplot(fig)
    st.markdown('''Durbin-Watson hypothesis test for autocorrelation, which can be a value belonged to these ranges:
        - $0 < val < 1$: the data has positive autocorrelation.
        - $1< val <3$: the data has no autocorrelation.
        - $3 < val < 4$: the data has negative autocorrelation.''')
    st.text('Durbin-Watson: ' + str(durbin_watson(acf_cal["Life Expectancy"])))
    st.markdown('''Conclusion:
        - Life expectancy has positive autocorrelation.
        - Life expectancy within this dataset is not cyclical.
        ### Detrending''')
    
    #-----------------Detrending-----------------
    temp = world.reset_index().rename(columns={'World':'Life Expectancy'})
    detr = signal.detrend(temp['Life Expectancy'])
    detr = pd.DataFrame({'LE detrend':detr},index = world.index)

    temp.set_index('Year', inplace=True)
    fig = plt.figure()
    plt.plot(temp["Life Expectancy"], label=temp.columns[0])
    plt.plot(detr, label=detr.columns[0])
    plt.xticks(rotation=45)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Life Expentancy", fontsize=12)
    plt.legend(fontsize=12)
    plt.title("Life Expectancy from 2000 to 2015", fontsize=18)
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)

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

def compare_vn_world(df_life):
    st.markdown('## Comparation of Life Expectancy between Viet Nam and the World ')
    st.markdown('### The mean life expectancy of Vietnam with the global average')

    # Calculate mean life expectancy for Vietnam and global
    global_mean_life_expectancy = df_life["Life expectancy"].mean()
    vietnam_mean_life_expectancy = df_life[df_life["Country"] == "Viet Nam"]["Life expectancy"].mean()

    # Create a bar plot with customized colors
    fig = plt.figure()
    plt.bar(["Global", "Vietnam"], [global_mean_life_expectancy, vietnam_mean_life_expectancy], color=["green", "red"])
    plt.title("Mean Life Expectancy: Global vs Vietnam")
    plt.xlabel("Country")
    plt.ylabel("Mean Life Expectancy")
    st.pyplot(fig)

    st.markdown('### The life expectancy ranking of Vietnam each year ')

    # Filter the data from the year 2000 onwards
    df_from_2000 = df_life[df_life['Year'] >= 2000]

    # Create an empty DataFrame to store the rankings
    df_ranking = pd.DataFrame(columns=['Year', 'Rank'])

    # Iterate over each year
    for year in df_from_2000['Year'].unique():
        df_year = df_from_2000[df_from_2000['Year'] == year]
        df_year_sorted = df_year.sort_values('Life expectancy', ascending=False)
        df_year_sorted.reset_index(drop=True, inplace=True)
        df_year_sorted.index = df_year_sorted.index + 1
        vietnam_rank = df_year_sorted[df_year_sorted['Country'] == 'Viet Nam'].index[0]
        df_ranking = df_ranking.append({'Year': year, 'Rank': vietnam_rank}, ignore_index=True)

    # Plot the ranking of Vietnam over the years
    fig = plt.figure()
    plt.plot(df_ranking['Year'], df_ranking['Rank'], marker='o')
    plt.title('Ranking of Vietnam in Life Expectancy')
    plt.xlabel('Year')
    plt.ylabel('Rank')
    plt.grid(True)
    st.pyplot(fig)

    st.markdown('### Compare the life expectancy of Vietnam with the top 5 countries with the highest life expectancy')

    df_avg_life_expectancy = df_life.groupby('Country')['Life expectancy'].mean().reset_index()
    df_avg_life_expectancy = df_avg_life_expectancy.sort_values('Life expectancy', ascending=False)
    top_5_countries = df_avg_life_expectancy.head(5)['Country'].tolist()
    selected_countries = ['Viet Nam'] + top_5_countries
    df_comparison = df_life[df_life['Country'].isin(selected_countries)]
    fig = plt.figure()
    sns.lineplot(data=df_comparison, x='Year', y='Life expectancy', hue='Country', marker='o')
    plt.title('Comparison of Life Expectancy: Vietnam vs Top 5 Countries')
    plt.xlabel('Year')
    plt.ylabel('Life Expectancy')
    plt.legend(loc='center left', bbox_to_anchor=(1, .5))
    st.pyplot(fig)

    st.markdown('### The life expectancy of Vietnam and Southeast Asian countries in the most recent 5 years')

    # Filter the data for Southeast Asian countries and the most recent 5 years
    southeast_asian_countries = ['Viet Nam', 'Thailand', 'Indonesia', 'Philippines', 'Malaysia', 'Singapore', 'Cambodia', 'Myanmar', 'Laos']
    df_southeast_asia = df_life[df_life['Country'].isin(southeast_asian_countries)]
    recent_years = df_southeast_asia['Year'].max() - 4
    df_recent = df_southeast_asia[df_southeast_asia['Year'] >= recent_years]

    # Plot the life expectancy of Vietnam and Southeast Asian countries in the most recent 5 years
    fig = plt.figure()
    plt.plot(df_recent[df_recent['Country'] == 'Viet Nam']['Year'], df_recent[df_recent['Country'] == 'Viet Nam']['Life expectancy'], marker='o', label='Vietnam')
    for country in southeast_asian_countries:
        if country != 'Viet Nam':
            plt.plot(df_recent[df_recent['Country'] == country]['Year'], df_recent[df_recent['Country'] == country]['Life expectancy'], marker = 'o', label=country)
    plt.title("Life Expectancy Comparison: Vietnam vs Southeast Asian Countries")
    plt.xlabel("Year")
    plt.ylabel("Life Expectancy")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    st.pyplot(fig)

    st.markdown('### The life expectancy rankings of Vietnam and other countries in Southeast Asia in 2015')

    # Filter the data for Southeast Asian countries in the year 2015
    df_sea_2015 = df_life[(df_life['Country'].isin(southeast_asian_countries)) & (df_life['Year'] == 2015)]
    # Sort the data by life expectancy in descending order
    df_sea_2015_sorted = df_sea_2015.sort_values('Life expectancy', ascending=False)
    # Get the rank of Vietnam in 2015
    vietnam_rank = df_sea_2015_sorted[df_sea_2015_sorted['Country'] == 'Viet Nam'].index[0] + 1
    # Create a bar plot for life expectancy rankings in Southeast Asia (2015)
    fig = plt.figure()
    sns.set_style('whitegrid')
    plt.bar(df_sea_2015_sorted['Country'], df_sea_2015_sorted['Life expectancy'], color=['blue' if country != 'Viet Nam' else 'red' for country in df_sea_2015_sorted['Country']])
    plt.xlabel('Country')
    plt.ylabel('Life Expectancy')
    plt.title('Life Expectancy Rankings in Southeast Asia (2015)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown('''
        #### Conclusions:
        - The mean life expectancy of Vietnam is higher than the global average.
        - From 2000, Vietnam's ranking in terms of life expectancy falls within the range of the top 47 to 59 countries. In 2004, Vietnam ranked 47th globally, which was the highest ranking among the years analyzed. However, in 2015, Vietnam dropped to the 59th position globally, which was the lowest ranking among the years analyzed. 
        - The life expectancy of Vietnam, based on statistics from 2000, is significantly lower compared to the top 5 countries with the highest average life expectancy in the world.
        - From 2011 to 2015, Vietnam maintained its second-highest position in terms of life expectancy among the countries in the Southeast Asian region. Singapore remained at the top of the region in terms of life expectancy.
    ''')

# -------------------------AR(1) model-------------------------


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
    feature_value = [0.529923, 0.211056, 0.163982,0.029546]


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
multivariate_analysis()
compare_vn_world(df)
timeseries_analysis()
# regression_analysis()
# solution()