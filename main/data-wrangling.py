import pandas as pd

df = pd.read_csv("Data/falcon9_launches.csv")
#df = pd.read_csv("Data/dataset_part_1.csv")
#print(df["LaunchSite"].value_counts())
#print(df["Orbit"].value_counts())
landing_outcomes = df["Outcome"].value_counts()
for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
#print(bad_outcomes)

landing_class = [0 if outcome in bad_outcomes else 1 for outcome in df["Outcome"]]
df['Class']=landing_class
print(df["Class"].mean())

df.to_csv("dataset_part_2.csv", index=False)
