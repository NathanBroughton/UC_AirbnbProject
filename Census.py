# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:41:46 2022

@author: NatBr
"""

import pandas as pd
import json
from census import Census
from us import states
import numpy as np

city = 'Washington D.C.'
with open('UC_AirbnbProject-main/tracts.txt', 'r') as f:
    areas = json.loads(f.read())
    
#print("Total number of cities =", len(areas.keys()))
#print("Area {} =".format(city), areas[city])

with open('UC_AirbnbProject-main/area_per_tract.txt', 'r') as f:
    sizes = json.loads(f.read())
#print("Total number of cities =", len(sizes.keys()))
#print("Sizes {} =".format(city), sizes[city])
keys = []
values = []
for key, value in sizes[city].items():
    keys.append(key)
    values.append(value)
sizes = pd.DataFrame({'Tracts':keys,'Sizes':values})

with open('UC_AirbnbProject-main/hotels_per_tract.txt', 'r') as f:
    hotels = json.loads(f.read())
#print("Total number of cities =", len(hotels.keys()))
#print("Hotels {} =".format(city), hotels[city])
keys = []
values = []
for key, value in hotels[city].items():
    keys.append(key)
    values.append(value)
hotels = pd.DataFrame({'Tracts':keys,'Number of Hotels':values})
#print(hotels)

with open('UC_AirbnbProject-main/center_distance_per_tract.txt', 'r') as f:
    centers = json.loads(f.read())
#print("Total number of cities =", len(hotels.keys()))
#print("Hotels {} =".format(city), hotels[city])
keys = []
values = []
for key, value in centers[city].items():
    keys.append(key)
    values.append(value)
center = pd.DataFrame({'Tracts':keys,'Distance to Center':values})


with open('UC_AirbnbProject-main/busstops_per_tract.txt', 'r') as f:
    busstops = json.loads(f.read())
#print("Total number of cities =", len(busstops.keys()))
#print("Busstops {} =".format(city), busstops[city])
keys = []
values = []
for key, value in busstops[city].items():
    keys.append(key)
    values.append(value)
busstops = pd.DataFrame({'Tracts':keys,'Bus Stops':values})
#print(busstops)

with open('UC_AirbnbProject-main/poi_per_tract.txt', 'r') as f:
    poi = json.loads(f.read())
#print("Total number of cities =", len(poi.keys()))
#print("Busstops {} =".format(city), poi[city])
keys = []
values = []
for key, value in poi[city].items():
    keys.append(key)
    values.append(value)
poi = pd.DataFrame({'Tracts':keys,'Point of Interests':values})
#print(poi)

with open('UC_AirbnbProject-main/airbnb_per_tract.txt', 'r') as f:
    airbnb = json.loads(f.read())
#print("Total number of cities =", len(airbnb.keys()))
#print("Busstops {} =".format(city), airbnb[city])
keys = []
values = []
for key, value in airbnb[city].items():
    keys.append(key)
    values.append(value)
airbnb = pd.DataFrame({'Tracts':keys,'Airbnb penetration':values})
#print(airbnb)



#print(len(areas[city]),len(sizes[city]),len(hotels[city]),len(busstops[city]))
df = pd.DataFrame({'Tracts': []})
df['Tracts'] = areas[city]
#df['Sizes'] = sizes[city]
#print(df.head())

df = df.merge(sizes, on='Tracts', how='left')
df = df.merge(hotels, on='Tracts', how='left')
df = df.merge(busstops, on='Tracts', how='left')
df = df.merge(poi, on='Tracts', how='left')
df = df.merge(center, on='Tracts', how='left')
df = df.merge(airbnb, on='Tracts', how='left')

print(df.head())
print(df.columns)
print('Shape: ', df.shape)
#df.to_csv('Data/test.csv')


c = Census("f4e210c1005f884bf6be7334f040ab9eea991c08")

#Name, Total Population, Race: AI,AS,B,HL,NH,T,W,O, GINI,bohemian: M, F, T, Young: M20,M21,M22-24,M25-29,M30-34, F20,F21,F22-24,F25-29,F30-34, Education: B,M,P,D, Poverty (number)+Income (total),Median household income, Median household value, Owner: O, T, No. of unemployed in labor force

fields = ('NAME','B01001_001E','B03002_005E','B03002_006E','B03002_004E','B03002_001E','B03002_007E', \
          'B03002_009E','B03002_003E','B03002_008E','B19083_001E', \
          'C24010_015E','C24010_051E','C24010_001E','B01001_008E','B01001_009E', \
          'B01001_010E','B01001_011E','B01001_012E','B01001_032E','B01001_033E', \
          'B01001_034E','B01001_035E','B01001_036E','B15003_022E','B15003_023E', \
          'B15003_024E','B15003_025E','B17001_002E','B17001_001E','B19013_001E', \
          'B25077_001E','B25002_002E','B25002_001E','B23025_005E')

va_census = c.acs5.state_county_tract(fields = fields,
                                      state_fips = states.DC.fips,
                                      county_fips = "*",
                                      tract = "*",
                                      year = 2020)

# Create a dataframe from the census data
va_df = pd.DataFrame(va_census)

#print(list(set(areas[city]).difference(va_df['NAME'].str.split(',').str[0])))

print(va_df.columns)
va_df['NAME'] = va_df['NAME'].str.split(',').str[0]

print(va_df['NAME'].duplicated().any())

#print(va_df['NAME'].str.split(',').str[0])
# Show the dataframe
print(va_df.head(2))
print('Shape: ', va_df.shape)

#va_df = va_df.set_index('NAME').join(df.set_index('Tracts'))
print(va_df.columns)
va_df = va_df.merge(df, left_on = ['NAME'],right_on = ['Tracts'] , how = 'inner')
print(va_df.columns)
va_df = va_df.drop_duplicates(subset = 'tract')

#Pop_dens
va_df['Population Density'] = va_df['B01001_001E']/va_df['Sizes']
va_df = va_df.drop(columns=['Sizes'])

#Race_div
va_df['totalN'] = va_df['B03002_008E'] + va_df['B03002_006E'] +va_df['B03002_004E'] + va_df['B03002_001E'] + \
    va_df['B03002_007E'] + va_df ['B03002_009E'] + va_df['B03002_003E'] + va_df['B03002_005E']
va_df['N'] = va_df['totalN']*(va_df['totalN']-1)
keys = ['B03002_005E','B03002_006E','B03002_004E','B03002_001E','B03002_007E','B03002_009E','B03002_003E','B03002_008E']
for i in range(len(keys)):
    key = "n{}".format(i)
    va_df[key] = va_df[keys[i]]*(va_df[keys[i]]-1)
va_df['n'] = va_df['n0']+va_df['n1']+va_df['n2']+va_df['n3']+va_df['n4']+va_df['n5']+va_df['n6']+va_df['n7']
va_df['Race Diversity Index'] = va_df['n']/va_df['N']

va_df = va_df.drop(columns=['B03002_005E','B03002_006E','B03002_004E','B03002_001E','B03002_007E','B03002_009E','B03002_003E','B03002_008E','n0','n1','n2','n3','n4','n5','n6','n7','n','totalN','N'])

#GINI
va_df = va_df.rename(columns={'B19083_001E':'Income Diversity Index'})

#Bohemian
df_Bohemian = pd.read_csv("Occupation.csv")

key = "United States!!Total!!Estimate"
column_sum = 0
for i in range(len(df_Bohemian[key])):
    column_sum += int(df_Bohemian[key][i].replace(',',''))
US = int(df_Bohemian[key][13].replace(',',''))/column_sum

va_df['Bohemian Index'] = ((va_df['C24010_015E']+ va_df['C24010_051E'])/va_df['C24010_001E'])/US

va_df = va_df.drop(columns=['C24010_015E','C24010_051E','C24010_001E'])

#Talent
va_df['Talent Index'] = (va_df['B15003_022E'] + va_df['B15003_023E'] + va_df['B15003_024E'] +
    va_df['B15003_025E'])/va_df['B01001_001E']

va_df = va_df.drop(columns=['B15003_022E','B15003_023E', 'B15003_024E','B15003_025E'])
#Young
va_df['Proportion of Young People'] = (va_df['B01001_008E'] + va_df['B01001_009E'] +
    va_df['B01001_010E'] + va_df['B01001_011E'] + va_df['B01001_012E'] + 
    va_df['B01001_032E'] + va_df['B01001_033E'] + va_df['B01001_034E'] + 
    va_df['B01001_035E'] + va_df['B01001_036E'])/va_df['B01001_001E']

va_df = va_df.drop(columns=['B01001_008E','B01001_009E', \
          'B01001_010E','B01001_011E','B01001_012E','B01001_032E','B01001_033E', \
          'B01001_034E','B01001_035E','B01001_036E'])

#Unemployed
va_df['Unemployment Ratio'] = va_df['B23025_005E']/va_df['B01001_001E']

va_df = va_df.drop(columns=['B23025_005E','B01001_001E'])
#Poverty
va_df['Poverty by Income Percentage'] = va_df['B17001_002E']/va_df['B17001_001E']

va_df = va_df.drop(columns = ['B17001_002E','B17001_001E'])

#Median Household Income
va_df = va_df.rename(columns={'B19013_001E':'Median Household Income'})

#Median Househod Value
va_df = va_df.rename(columns={'B25077_001E':'Median Household Value'})

#Occupied
va_df['Proportion of Owner Occupied Residences'] = va_df['B25002_002E']/va_df['B25002_001E']

va_df = va_df.drop(columns=['B25002_002E','B25002_001E'])

print(va_df.head(2))
print('Shape: ', va_df.shape)
va_df = va_df.fillna(0)
va_df = va_df.replace(-666666666,np.nan)
va_df = va_df.dropna()
va_df.to_csv('Data3/WashtingtonDC.csv')
