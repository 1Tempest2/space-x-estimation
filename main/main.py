import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import requests
import datetime

#Global variables
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []


def getBoosterVersion(data):
    for x in data['rocket']:
       if x:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])
def getLaunchSite(data):
    for x in data['launchpad']:
       if x:
         response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
         Longitude.append(response['longitude'])
         Latitude.append(response['latitude'])
         LaunchSite.append(response['name'])
def getPayloadData(data):
    for load in data['payloads']:
       if load:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])

def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])

#making it so everything is displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#spacex_url="https://api.spacexdata.com/v4/launches/past"
spacex_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
response = requests.get(spacex_url)
print(response.status_code)

dataset = pd.json_normalize(response.json())
dataset = dataset[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]
dataset = dataset[dataset['cores'].map(len) == 1]
dataset = dataset[dataset['payloads'].map(len) == 1]

dataset['cores'] = dataset['cores'].map(lambda x : x[0])
dataset['payloads'] = dataset['payloads'].map(lambda x : x[0])

dataset["date"] = pd.to_datetime(dataset['date_utc']).dt.date
dataset = dataset[dataset["date"] < datetime.date(2020, 11, 13)]

#Getting data
getBoosterVersion(dataset)
getLaunchSite(dataset)
getPayloadData(dataset)
getCoreData(dataset)

#Collecting it into a dictionary
launch_dict = {'FlightNumber': list(dataset['flight_number']),
'Date': list(dataset['date']),
'BoosterVersion':BoosterVersion,
'PayloadMass':PayloadMass,
'Orbit':Orbit,
'LaunchSite':LaunchSite,
'Outcome':Outcome,
'Flights':Flights,
'GridFins':GridFins,
'Reused':Reused,
'Legs':Legs,
'LandingPad':LandingPad,
'Block':Block,
'ReusedCount':ReusedCount,
'Serial':Serial,
'Longitude': Longitude,
'Latitude': Latitude}

df = pd.DataFrame(launch_dict)
#print(df.head())

#Filtering to only include falcon-9 rockets
df_falcon9 = df[df["BoosterVersion"] == "Falcon 9"].copy()
df_falcon9.loc[:, 'FlightNumber'] = list(range(1, df_falcon9.shape[0] + 1))
df_falcon9["PayloadMass"] = df_falcon9["PayloadMass"].fillna(df_falcon9["PayloadMass"].mean())

#print(df_falcon9.count())





