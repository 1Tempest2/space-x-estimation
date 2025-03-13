import csv, sqlite3
import prettytable as pt
import pandas as pd

pt.prettytable.DEFAULT = pt.TableStyle.SINGLE_BORDER

con = sqlite3.connect("Databases/spacex.db")
cur = con.cursor()

df = pd.read_csv("Data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False, method="multi")


#cur.execute("SELECT * FROM SPACEXTBL")
#print(prettytable.from_db_cursor(cur))


cur.execute("SELECT DISTINCT Launch_Site from SPACEXTBL")
#print(pt.from_db_cursor(cur))

cur.execute("SELECT * FROM SPACEXTBL WHERE Launch_Site LIKE 'CCA%' LIMIT 100")
#print(pt.from_db_cursor(cur))
cur.execute("SELECT SUM(PAYLOAD_MASS__KG_) FROM SPACEXTBL WHERE CUSTOMER LIKE 'NASA (CRS)'")
#print(pt.from_db_cursor(cur)) #45596
cur.execute("SELECT AVG(PAYLOAD_MASS__KG_) FROM SPACEXTBL WHERE Booster_Version LIKE 'F9 v1.1'")
#print(pt.from_db_cursor(cur)) 2928.4
cur.execute("SELECT MIN(DATE) FROM SPACEXTBL WHERE Landing_Outcome LIKE 'Success'")
#print(pt.from_db_cursor(cur)) 2018-07-22
cur.execute("SELECT DISTINCT Booster_Version FROM SPACEXTBL WHERE Landing_Outcome LIKE 'Success' AND PAYLOAD_MASS__KG_ BETWEEN 4000 AND 6000")
#print(pt.from_db_cursor(cur))
cur.execute("SELECT COUNT(CASE WHEN Mission_Outcome LIKE 'Success%' THEN 1 END) AS successfull_outcome,COUNT(CASE WHEN Mission_Outcome NOT LIKE 'Success%' THEN 1 END) AS failure_outcome FROM SPACEXTBL")
#print(pt.from_db_cursor(cur)) 100 1
cur.execute("SELECT DISTINCT Booster_Version FROM SPACEXTBL WHERE PAYLOAD_MASS__KG_ = (SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTBL)")
#print(pt.from_db_cursor(cur))
cur.execute("SELECT substring(DATE, 6,2), LANDING_OUTCOME, BOOSTER_VERSION, LAUNCH_SITE FROM SPACEXTBL WHERE SUBSTRING(DATE,0,5) = '2015' AND LANDING_OUTCOME LIKE 'Failure (drone ship)' ")
#print(pt.from_db_cursor(cur)) 2
cur.execute("SELECT LANDING_OUTCOME, COUNT(LANDING_OUTCOME) AS COUNT FROM SPACEXTBL GROUP BY LANDING_OUTCOME HAVING DATE BETWEEN '2010-06-04' AND '2017-03-20' ORDER BY DATE DESC ")
#print(pt.from_db_cursor(cur))
con.close()