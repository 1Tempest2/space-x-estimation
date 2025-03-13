import pandas as pd
import folium
from folium import Marker
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from folium.features import DivIcon

df = pd.read_csv("Data/spacex_launch_geo.csv")
spacex_df = df[['Launch_Site', 'Lat', 'Long', 'Class']]
launch_sites_df = spacex_df.groupby(['Launch_Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch_Site', 'Lat', 'Long']]

nasa_coordinate = [29.559684888503615, -95.0830971930759]
launch_sites_map = folium.Map(location=nasa_coordinate, zoom_start=11, tiles='OpenStreetMap')
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
launch_sites_map.add_child(circle)

for row in launch_sites_df.itertuples():
    name, lat, long = row.Launch_Site, row.Lat, row.Long
    launch_sites_map.add_child(folium.Circle([lat, long], radius=1000, color='#d7565b', fill=True))
    launch_sites_map.add_child(folium.Marker([lat, long], icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d7565b;"><b>%s</b></div>' % name )))
marker_cluster = MarkerCluster()
launch_sites_map.add_child(marker_cluster)

for row in spacex_df.itertuples():
    lat, long, Class = row.Lat, row.Long, row.Class

    # Set color based on class
    color = "#d7565b" if Class == 0 else "#00FF00"

    # Create the marker with cleaner formatting
    marker = folium.Marker(
        [lat, long],
        icon=folium.Icon(color='gray', icon_color=color)
    )

    marker_cluster.add_child(marker)
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN', 
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

launch_sites_map.add_child(mouse_position)

launch_sites_map.show_in_browser()