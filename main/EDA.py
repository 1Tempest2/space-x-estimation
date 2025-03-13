import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prettytable as pt

def df_to_prettytable(df):
    table = pt.PrettyTable()
    table.field_names = df.columns.tolist()
    for row in df.values:
        table.add_row(row)
    return table

#pd.set_option('display.max_rows', 500)
df = pd.read_csv("Data/dataset_part_2.csv")
#print(df.head())
print(df.columns)

#finding correlation between data
#   sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
#plt.show()

#AS the flight number increased so did the success rate of the landing of the first stage
#sns.catplot(y="LaunchSite",
#            x="FlightNumber",
#            hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("LaunchSite",fontsize=20)
#plt.show()

#sns.catplot(y="LaunchSite",
#            x="PayloadMass",
#            hue="Class", data=df, aspect = 5)
plt.xlabel("PayloadMass",fontsize=20)
plt.ylabel("LaunchSite",fontsize=20)
#plt.show()

success_rates = df.groupby("Orbit")["Class"].mean().reset_index()
plt.figure(figsize=(12, 6))
#sns.barplot(x="Class", y="Orbit", data=success_rates, palette="viridis")
plt.xlabel("Success Rate", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
#plt.show()

# sns.scatterplot(
#     x="FlightNumber",
#     y="Orbit",
#     hue="Class",
#     data=df,
#     palette="viridis"
# )
plt.xlabel("Flight N0", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
#plt.show()
# sns.scatterplot(
#     x="PayloadMass",
#     y="Orbit",
#     hue="Class",
#     data=df,
#     palette="viridis"
# )
# plt.xlabel("PayloadMass", fontsize=14)
# plt.ylabel("Orbit Type", fontsize=14)
# plt.show()

features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
#print(df_to_prettytable(features))
features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])
numeric_columns = features_one_hot.select_dtypes(include=['int64', 'float64']).columns
features_one_hot[numeric_columns] = features_one_hot[numeric_columns].astype('float64')
#print(df_to_prettytable(features_one_hot))
print(len(features_one_hot.columns))
features_one_hot.to_csv('dataset_part_3.csv', index=False)
