"""
#####################################################################################################
Note: 
from line 1-180, it is for data collection and cleaning
every extracted variable will be put into covid1 for further analysis
After line 181, it is for data analysis
variables are being calculated to come up with 4 sub-scores, [Pandemic, predictions, access, Diversity]
and the total score is being calculated at the end
#####################################################################################################
"""
'''
extracted from 'Bing COVID-19 Tracker' 
www.bing.com/covid

### to extract the lastest covid data by country
### data included:
iso  country  TodayConfirmed  TodayConfirmedChange  14DaysConfirmed      Death     Recovered
'''
import pandas as pd
# read csv file from github
url1 = 'https://github.com/microsoft/Bing-COVID-19-Data/blob/master/data/Bing-COVID19-Data.csv?raw=true'
df1 = pd.read_csv(url1, index_col=0)

# filter out history data
real_today = pd.Timestamp.today()
today = real_today - pd.Timedelta("3 days") # get today date (most update: 2 days before)
today = str(today.strftime('%m/%d/%Y'))
before = real_today - pd.Timedelta("10 days") # get the date one weeks ago
fourteen = str(before.strftime('%m/%d/%Y'))
df_today = df1[df1['Updated'] == today] # filter out the today's data
df_7days = df1[df1['Updated'] == fourteen] # filter out the data wih seven days ago

# simplify data
df_today.drop(df_today.loc[df_today['AdminRegion1'].notnull()].index, inplace=True) # drop the states data in a country
df_today = df_today.drop_duplicates(subset=['ISO3',"Country_Region","Latitude","Longitude"], keep='first') # drop duplicates
df_7days.drop(df_7days.loc[df_7days['AdminRegion1'].notnull()].index, inplace=True)
df_7days = df_7days.drop_duplicates(subset=['ISO3',"Country_Region","Latitude","Longitude"], keep='first')

# delete countries that are not in both df_today and df_14days
excess = list(set(df_today["Country_Region"]).symmetric_difference(df_7days["Country_Region"]))
for i in range(len(excess)):
    if excess[i] in df_today.values: #if in today, delete (not in 14days)
        df_today = df_today[df_today['Country_Region'] != excess[i]]
    if excess[i] in df_7days.values: #if in 7days, delete (not in today)
        df_7days = df_7days[df_7days['Country_Region'] != excess[i]]       

df_today = df_today.set_index('ISO3',drop=False)
df_7days = df_7days.set_index('ISO3',drop=False)

# open a new dataframe to store data
data1 = pd.DataFrame()
# insert columns to new dataframe
data1['iso']=df_today['ISO3'] # iso_code
#set index
data1 = data1.set_index('iso')
data1['Country']=df_today['Country_Region'] # countries
data1['TodayConfirmed']=df_today['Confirmed'] #total confirmed
data1['TodayConfirmedChange']=df_today['ConfirmedChange'] #today confirmed 
data1['Death']=df_today['Deaths'] #total deaths
data1['Recovered']=df_today['Recovered'] #total recovered
data1.insert(loc=5, column="7DaysConfirmed", value=df_7days['Confirmed']) #1 weeks before total confirmed

'''
extracted from Data on COVID-19 (coronavirus) vaccinations by Our World in Data
https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/

### to extract the latest vaccineation data by country
### vaccine data will be added to the covid dataframe
### data included:
people_vaccinated  people_vaccinated_per_hundred
'''
# read csv file from github
url = 'https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv?raw=true'
df2 = pd.read_csv(url, index_col=0)
# read covid csv file for combination for two datasets 
covid1 = data1

# remove past data, only keep the newest
df2 = df2.drop_duplicates(subset=["iso_code"], keep='last') # drop duplicates

# set index for combination
df2 = df2.set_index('iso_code')

# insert vaccination data to covid dataset
covid1.insert(loc=6, column="people_vaccinated", value=df2["people_vaccinated"])
covid1.insert(loc=7, column="people_vaccinated_per_hundred", value=df2["people_vaccinated_per_hundred"])

# remove the countries which doesn't have vaccination data
covid1 = covid1.dropna(axis=0, subset=['people_vaccinated'])

# calculate total population (vaccination/vaccated per hundred*100)
covid1['Population'] = covid1["people_vaccinated"] / covid1['people_vaccinated_per_hundred'] * 100

covid1 = covid1[covid1['people_vaccinated_per_hundred'] !=0]

"""
extracted from Institute for Health Metrics and Evaluation
Dataset name:COVID-19 Mortality, Infection, Testing, Hospital Resource Use, and Social DistancingProjections
http://www.healthdata.org/covid

### to extract the predicted data
### data included:
estimated infections from today (day 1) to day 7
"""

# red the csv file
df3 = pd.read_csv(r'reference_hospitalization_all_locs.csv')

# create a new dataframe to store data
data2 = pd.DataFrame(df3['location_name'].unique())

# filter out history data
for i in range(7): # choose the next seven days for prediction
    date =  pd.Timestamp.today()
    date = date + pd.Timedelta(i,"days")
    date = str(date.strftime('%Y-%m-%d'))
    add = df3[df3['date'] == date]["est_infections_mean"]
    add.reset_index(drop=True, inplace=True)
    data2["day "+str(i)] = add

data2.columns = ["Country", 'day 0', 'day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'day 6'] # add the columns to the dataframe
data2 = data2.sort_values(by=["Country"])

# modify some row names
data2.Country[161]="Hong Kong SAR"
data2.Country[39]="South Korea"
data2.Country[2]="Taiwan"
data2.Country[69]="United States"
#Hong Kong Special Administrative Region of China 161
#Republic of Korea 39
#Taiwan (Province of China) 2
#United States of America 69

# modify the index so to simplify the combination process
data2 = data2.set_index('Country')
covid1 = covid1.reset_index()
covid1 = covid1.set_index("Country")

# add the data to the covid1 dataframe
covid1["day 0"] = data2["day 0"]
covid1["day 1"] = data2["day 1"]
covid1["day 2"] = data2["day 2"]
covid1["day 3"] = data2["day 3"]
covid1["day 4"] = data2["day 4"]
covid1["day 5"] = data2["day 5"]
covid1["day 6"] = data2["day 6"]

# remove the countries which doesn't have vaccination data
covid1 = covid1.dropna(axis=0, subset=['day 0','iso'])

"""
extracted from The Travel & Tourism Competitiveness Report 2019

### modified in csv to take useful data only
### data included:
'International Openness', 'Air transport infrastructure',
       'Ground and port infrastructure', 'Tourist service infrastructure',
       'Prioritization of Travel & Tourism', 'Natural and cultural resources ',
       'Environmental sustainability', 'Price competitiveness',
       'Safety and security', 'TTCI'
"""

# read csv file
df4 = pd.read_csv(r'modifiedTTCI.csv')
df4 = df4.set_index("Country")
# add columns to covid1 dataframe
for i in ['International Openness', 'Air transport infrastructure',
       'Ground and port infrastructure', 'Tourist service infrastructure',
       'Prioritization of Travel & Tourism', 'Natural and cultural resources ',
       'Environmental sustainability', 'Price competitiveness',
       'Safety and security', 'TTCI']:
    covid1[i] = df4[i] # add columns

#drop away the null cells
covid1 = covid1.dropna(axis=0, subset=['TTCI','iso'])
covid1 = covid1.reset_index() # reset index
"""
##################################################################################
END OF DATA COLLECTION
START ANALYZING DATA
##################################################################################
"""

"""
1. Pandemic
It is calculated by four parts:
PercentageChange = -(0.8 * (TodayConfirmed - 7DaysConfirmed) / TodayConfirmed) + 0.2 * (TodayConfirmedChange / TodayConfirmed)
DeathOverConfirmed  = -Death/TodayConfirmed
ConfirmOverPopulation = -TodayConfirmed / Population
RecoverOverConfirm = Recovered / TodayConfirmed

using the following variables:
TodayConfirmed	TodayConfirmedChange	Death	Recovered	7DaysConfirmed	Population
"""

# open a new dataframe for data storage
analyze1 = pd.DataFrame(covid1["Country"])

# calculate PercentageChange
PercentageChange = -0.8*((covid1["TodayConfirmed"]-covid1["7DaysConfirmed"])/covid1["TodayConfirmed"]) - 0.2*covid1["TodayConfirmedChange"]/covid1["TodayConfirmed"]
analyze1["PercentageChange"] = PercentageChange # add to the dataframe
# calculate DeathOverConfirmed
DeathOverConfirmed = -covid1["Death"]/covid1["TodayConfirmed"]
analyze1["DeathOverConfirmed"] = DeathOverConfirmed # add to the dataframe
# calculate ConfirmOverPopulation
ConfirmOverPopulation = -covid1["TodayConfirmed"]/covid1["Population"]
analyze1["ConfirmOverPopulation"] = ConfirmOverPopulation # add to the dataframe
# calculate RecoverOverConfirm
RecoverOverConfirm = covid1["Recovered"]/covid1["Population"]
analyze1["RecoverOverConfirm"] = RecoverOverConfirm # add to the dataframe

# copy the data
standard1 = analyze1.copy()
  
# apply standardaization 
for column in ["PercentageChange","DeathOverConfirmed","ConfirmOverPopulation","RecoverOverConfirm"]:
    standard1[column] = ((analyze1[column] - analyze1[column].mean()) / analyze1[column].std())

### deal with missing values
standard1['RecoverOverConfirm'].fillna(value=standard1['RecoverOverConfirm'].mean(), inplace=True)

# calculate the total pandemic score
# weightings are 1.5 in %change, 0.5 in death/confirm, 1 in confirm/population and 1 in recover/confirm
pandemic = 1.5*standard1["PercentageChange"]+0.5*standard1["DeathOverConfirmed"]+standard1["ConfirmOverPopulation"]+standard1["RecoverOverConfirm"]
standard1["pandemic"]=pandemic/4

# apply normalization to a 0-100 scale
standard1["pandemic"] = 100*(standard1["pandemic"] - standard1["pandemic"].min()) / (standard1["pandemic"].max() - standard1["pandemic"].min())
# modify the max and min value (since the gap is too large)
standard1.at[(standard1[standard1["pandemic"]==100].index),'pandemic'] =  standard1["pandemic"].mean() + 2*standard1["pandemic"].std()
standard1.at[(standard1[standard1["pandemic"]==0].index),'pandemic'] =  standard1["pandemic"].mean() - 2*standard1["pandemic"].std()

# add to the covid1 dataframe with 10 pt increase to balance the mean/sd
covid1["Pandemic"] = standard1["pandemic"]+10


"""
2. Prediction
It is calculated by two parts:
vaccination per 100 people
prediction from external website (extract the negative slope to find out the trend)

"""
# import numpy and sklearn for regression
import numpy as np
from sklearn.linear_model import LinearRegression
prediction = []

#  be the n-th day
x = np.array(range(1,8)).reshape((-1, 1))
# extract the n-th day prediction of infected case 
predict = covid1[['day 0', 'day 1',
       'day 2', 'day 3', 'day 4', 'day 5', 'day 6']]
for i in range(len(predict)):
    y = np.array(predict.loc[i]) # turn every countries' data to numpy array
    model = LinearRegression().fit(x, y) # fits the model to a straight line
    prediction.append(-float(model.coef_)) # negative the slope with a 

# 
analyze2 = pd.DataFrame(covid1[["Country","people_vaccinated_per_hundred"]])
analyze2["slope"] = prediction/covid1["day 0"]

# copy the data
standard2 = analyze2.copy()
  
# apply standardaization techniques 
for column in ["people_vaccinated_per_hundred","slope"]:
    standard2[column] = ((analyze2[column] - analyze2[column].mean()) / analyze2[column].std())

# apply weighting on vaccine (1) and slope (2)
predictions = standard2["people_vaccinated_per_hundred"] + 2*standard2["slope"]
standard2["predictions"] = predictions/3
# normalize the data
standard2["predictions"] = 100*(standard2["predictions"] - standard2["predictions"].min()) / (standard2["predictions"].max() - standard2["predictions"].min())
# modify the max and min value (since the gap is too large)
standard2.at[(standard2[standard2["predictions"]==100].index),'predictions'] =  standard2["predictions"].mean() + 2*standard2["predictions"].std()
standard2.at[(standard2[standard2["predictions"]==0].index),'predictions'] =  standard2["predictions"].mean() - 2*standard2["predictions"].std()

# add to the covid1 dataframe with a division of 1.2, and add 5 to balance the mean and sd
standard2["predictions"] = (standard2["predictions"]/1.4)+20
covid1['predictions'] = standard2["predictions"]

"""
3. Accessibility
It is added by:
International Openness  Air transport infrastructure    Ground and port infrastructure  Tourist service infrastructure  Prioritization of Travel & Tourism
of the country
"""

# open a new dataframe to store data
analyze3 = pd.DataFrame(covid1["Country"])
# add up all variables
analyze3["access"] = covid1["International Openness"] + covid1["Air transport infrastructure"] + covid1["Ground and port infrastructure"] + covid1["Tourist service infrastructure"] + covid1["Prioritization of Travel & Tourism"]

# copy the data
standard3 = analyze3.copy()
# normalization
standard3["access"] = 100*(analyze3["access"] - analyze3["access"].min()) / (analyze3["access"].max() - analyze3["access"].min())
# modify the max and min value (since the gap is too large)
standard3.at[(standard3[standard3["access"]==100].index),'access'] =  standard3["access"].mean() + 2*standard3["access"].std()
standard3.at[(standard3[standard3["access"]==0].index),'access'] =  standard3["access"].mean() - 2*standard3["access"].std()

# add to the covid1 dataframe with a division of 2.5, with an addition of 40 to balance the mean and sd
standard3["access"] = standard3["access"]/2.5+40
covid1["access"] = standard3["access"]


"""
4. Diversity
It is added by:
Natural and cultural resources      Environmental sustainability
of the country
"""
# open a new dataframe
analyze4 = pd.DataFrame(covid1["Country"])
# add up to get the diversity score with a weighting of 2 to 1
analyze4["Diversity"] = 2*covid1["Natural and cultural resources "] + covid1["Environmental sustainability"]

# add to the covid1 dataframe with a multiplication of 3.5, with an addition of 25 to balance the mean and sd
covid1["Diversity"] = analyze4["Diversity"]*3.5+25


"""
5. Total score
It is calculated by the following formula:
score = ((2 * pandemic + 2 * prediction + accessibility + diversity) / 6) * 2 - 50

"""
#apply the formula
covid1['Total'] = ((2*(covid1["Pandemic"] + covid1["predictions"]) + covid1["access"] + covid1["Diversity"])/6)*2-55

# reset the index
covid1 = covid1.set_index("Country", drop=True)
covid1.sort_values(by=['Total'], inplace=True, ascending=False)
print(covid1[["Pandemic","predictions","access","Diversity","Total"]])
print(covid1.describe())

# export to a csv/json file
covid1.to_json(r'covid.json', orient="index")
 