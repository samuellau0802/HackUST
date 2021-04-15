# HackUST
### 1. data_collection+analysis.py 
This code is written for data collection, cleaning and analysis
The data is extracted from several external links, and then exported to a json file

Here are the data that is extracted from external links:
1. 'Bing COVID-19 Tracker' 
www.bing.com/covid
2. 'Data on COVID-19 (coronavirus) vaccinations by Our World in Data' 
https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/
3. 'Institute for Health Metrics and Evaluation COVID-19 Mortality, Infection, Testing, Hospital Resource Use, and Social DistancingProjections
http://www.healthdata.org/covid
(Note: It is stored in reference_hospitalization_all_locs.csv)
4. The Travel & Tourism Competitiveness Report 2019
https://reports.weforum.org/travel-and-tourism-competitiveness-report-2019/overall-results/
(Note: It is stored in modifiedTTCI.csv)

### 2. modifiedTTCI.csv
The data of Travel & Tourism Competitiveness Index with modification. It will be imported to the py file.

### 3. reference_hospitalization_all_locs.csv
The prediction of the infected data of each country. It will be imported to the py file.

### 4. covid.json
It is the output of the program. The json file will be stored in a database. 
