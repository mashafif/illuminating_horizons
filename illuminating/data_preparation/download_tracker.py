import sqlite3
import pandas
import numpy

#ID of points
points = pandas.read_csv('illuminating/data_preparation/CA_locations.csv')
P = numpy.arange(len(points))
Y = numpy.arange(1999,2024)
print(Y)
conn = sqlite3.connect('download_tracker.sql')

L = []
for y in Y:
    for p in P:
        f = f'{y}_{p}_radiance.csv'
        l = [f,0]
        L.append(l)

L = pandas.DataFrame(L)
L.columns=['FILE','DONE']
print(L)

# Write the DataFrame to the SQLite database
L.to_sql('FILES', conn, index=True, if_exists='replace')
