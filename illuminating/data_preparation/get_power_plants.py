import pandas as pd
import os
import sqlite3

def get_uspvdb(state:str="CA"):
    # Load the CSV file into a Pandas DataFrame
    csv_file_path = '/Users/kaorukure/code/mashafif/illuminating_horizons/raw_data/uspvdb_v2_0_20240801.csv'
    df = pd.read_csv(csv_file_path)
    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    # Write the DataFrame to the SQLite database
    df.to_sql('power_plants', conn, index=False, if_exists='replace')
    # Query
    query = f"SELECT ylat, xlong, p_area,p_year,p_cap_dc FROM power_plants WHERE p_state = '{state}'"
    result_df = pd.read_sql_query(query, conn)
    return result_df

if __name__ == "__main__":
    print(get_uspvdb("CA"))
    root_path = os.path.dirname(os.path.dirname(__file__))
    print(root_path)
