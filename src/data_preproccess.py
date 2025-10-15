## 1. Data Downloading
import kagglehub
from pathlib import Path
import shutil

path = kagglehub.dataset_download("hrishitpatil/flight-data-2024")
print("Downloaded to:", path)

data_folder = Path('../data')
data_folder.mkdir(exist_ok=True)

for f in Path(path).glob('*'):
    shutil.copy(f, data_folder / f.name)

## 2. Data Loading and Initial Exploration
import pandas as pd

df = pd.read_csv("../data/flight_data_2024.csv")
print(df.head())

## 3. Missing Values Check
df.isnull().sum()

# Note: Cancellation code will be dropped because of major missing values
df = df.drop(columns=["cancellation_code"])
print(df.columns)

# Note: dropping rows with missing observations - 110 thousand for 7 million observations is a small number
print("Before dropping:", len(df))
df = df.dropna()
print("After dropping:", len(df))

## 4. Noise in Data Check
from matplotlib import pyplot as plt

plt.hist(df["arr_delay"], range=(-50, 300), bins=200)
plt.show()

# Modelling the relation between arr_delay and dep_delay
import seaborn as sns

sns.jointplot(df, x="arr_delay", y="dep_delay")

# NOTE: Given that the relation between dep_delay and arr_delay stays mostly linear, we can assume that even the largest delays can be used in our models, because they happen in real life and are not simply noise in the data. Dropping them from our dataset could lead to fitting on artificial data.

## 5. Duplicate Check
copy_df = df.drop_duplicates()
print("Duplicates removed:", len(copy_df) - len(df))

## 6. Logical Data Pruning
def logical_checks(df):
    issues = {}

    issues['year_invalid'] = df[~df['year'].between(1900, 2100)]
    issues['month_invalid'] = df[~df['month'].between(1, 12)]
    issues['day_of_month_invalid'] = df[~df['day_of_month'].between(1, 31)]
    issues['day_of_week_invalid'] = df[~df['day_of_week'].between(1, 7)]

    for col in ['crs_dep_time','dep_time','wheels_off','wheels_on','crs_arr_time','arr_time']:
        issues[f'{col}_invalid'] = df[~df[col].between(0, 2400)]

    for col in ['taxi_out','taxi_in','crs_elapsed_time','actual_elapsed_time','air_time',
                'distance','carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay']:
        if col == 'distance':
            issues[f'{col}_invalid'] = df[df[col] < 0]
        else:
            issues[f'{col}_invalid'] = df[df[col] < 0]

    issues['cancelled_invalid'] = df[~df['cancelled'].isin([0,1])]
    issues['diverted_invalid'] = df[~df['diverted'].isin([0,1])]

    return {k:v for k,v in issues.items() if not v.empty}

problems = logical_checks(df)
for k, v in problems.items():
    print(k, len(v))

## 7. Checking the Amount of Unique Values in Each Column
unique_counts = df.nunique()
print(unique_counts)

unique_counts.plot(kind='bar', title='Histogram of Unique Values')
plt.xlabel('Columns')
plt.ylabel('Number of Unique Values')
plt.show()

## 8. Adding Columns for Classification Task Purposes
# We will assume that in terms of departing a plane is delayed if its delay is bigger than 15 minutes, because it's the most commonly assumed value.
# In terms of arrival delay, we decided that the plane is delayed on arrival if the delay is bigger than 0. We made that assumption because people often take transfer flights and in that case any delay on arrival can be disastrous for the transfer, so it makes sense to propose such a distinction.
# NOTE: In the dataset a delay can be negative if the plane arrives before the scheduled time.
df['is_dep_delayed'] = df['dep_delay'] > 15
df['is_arr_delayed'] = df['arr_delay'] > 0

print(df[['dep_delay', 'is_dep_delayed', 'arr_delay', 'is_arr_delayed']].head())

df['is_dep_delayed'].value_counts().plot(kind='pie', title='Proportion of Departure Delays', autopct='%1.1f%%')
plt.ylabel('')
plt.show()

df['is_arr_delayed'].value_counts().plot(kind='pie', title='Proportion of Arrival Delays', autopct='%1.1f%%')
plt.ylabel('')
plt.show()

print("Counts for is_dep_delayed:")
print(df['is_dep_delayed'].value_counts())

print("\nCounts for is_arr_delayed:")
print(df['is_arr_delayed'].value_counts())

print("Proportion for is_dep_delayed:")
print(df['is_dep_delayed'].value_counts(normalize=True))

print("\nProportion for is_arr_delayed:")
print(df['is_arr_delayed'].value_counts(normalize=True))

## 9. Changing Columns to Correct Data Type
# fl_date to datetime
df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")
print(df["fl_date"].dtype)
print(df["fl_date"][0])

# crs_dep_time to time
df["crs_dep_time"] = pd.to_datetime(df["crs_dep_time"], format='%H%M', errors="coerce").dt.time
print(df["crs_dep_time"].dtype)
print(df["crs_dep_time"][0])

# dep_time to time
df["dep_time"] = pd.to_datetime(df["dep_time"], format='%H%M', errors="coerce").dt.time
print(df["dep_time"].dtype)
print(df["dep_time"][0])

# arr_time to time
df["arr_time"] = pd.to_datetime(df["arr_time"], format='%H%M', errors="coerce").dt.time
print(df["arr_time"].dtype)
print(df["arr_time"][0])

# crs_arr_time to time
df["crs_arr_time"] = pd.to_datetime(df["crs_arr_time"], format='%H%M', errors="coerce").dt.time
print(df["crs_arr_time"].dtype)
print(df["crs_arr_time"][0])

## 10. Selecting Columns for Usage in ML Models
selected_columns = [
    'month', 'day_of_month', 'day_of_week', 'op_unique_carrier', 'origin',
    'origin_city_name', 'origin_state_nm', 'dest', 'dest_city_name',
    'dest_state_nm', 'dep_time', 'distance'
]

print("Selected columns:", selected_columns)

## 11. Saving the Preprocessed Data
df.to_csv("../data/preprocessed_flight_data.csv")