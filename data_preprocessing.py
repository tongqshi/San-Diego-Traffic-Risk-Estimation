import pandas as pd

def convert_street_suffix(x):
    ''''''
    street_abr = {'ST': 'STREET',
              'AVE': 'AVENUE',
              'AV': 'AVENUE',
              'RD': 'ROAD',
              'BL': 'BOULEVARD',
              'BLVD': 'BOULEVARD',
              'DR': 'DRIVE',
              'WY': 'WAY',
              'PY': 'PARKWAY',
              'PL': 'PLACE',
              'CT': 'COURT',
              'HY': 'HIGHWAY',
              'LN': 'LANE',
              'PLZ': 'PLAZA',
              '': 'CIRCLE',
              'ML': 'MALL',
              '': 'RAMP',
              '': 'WALK',
              '': 'TERRACE',
              'RW': 'ROW',
              '': 'DRIVEWAY',
              '': 'TRAIL',
              '': 'SQUARE',
              '': 'COVE',
              'PT': 'POINT',
              '': 'EXT ST',
              '': 'GLEN',
              '': 'BRIDGE',
              '': 'AVE',
              '': 'PASEO',
              '': 'ALLEY',
              '': 'PATH',
              '': 'KNOLLS',
              '': 'LIGHTS',
              '': 'CRESCENT'
              }
    if x[-1] in street_abr:
        x[-1] = street_abr.get(x[-1])
    x = ' '.join(x)
    return x

collision_reports = pd.read_csv('pd_collisions_details_datasd.csv')
traffic_counts = pd.read_csv('traffic_counts_datasd.csv')

# Change dates to month-year format
collision_reports['date_time'] = pd.to_datetime(collision_reports['date_time']).dt.to_period('m')
traffic_counts['date_count'] = pd.to_datetime(traffic_counts['date_count']).dt.to_period('m')

# Get rid of extraneous data
collision_reports = collision_reports.loc[:, ['date_time', 'address_road_primary', 'address_sfx_primary', 'address_name_intersecting', 'address_sfx_intersecting', 'veh_type', 'injured', 'killed']]
traffic_counts = traffic_counts.loc[:, ['date_count', 'street_name', 'total_count']]

# Abbrieviate all street suffixes // items without entry do not have abbrieviation

# Split street_name col by ' ', convert second element, put back in place
traffic_counts['street_name'] = traffic_counts['street_name'].str.split(' ')
traffic_counts['street_name'] = traffic_counts['street_name'].apply(convert_street_suffix)


# Merge collision report street names into single column
collision_reports['street_name_primary'] = collision_reports['address_road_primary'] + ' ' + collision_reports['address_sfx_primary']
collision_reports['street_name_primary'] = collision_reports['street_name_primary'].str.strip()
collision_reports.drop(columns = ['address_road_primary', 'address_sfx_primary'], inplace = True)

collision_reports['street_name_intersecting'] = collision_reports['address_name_intersecting'] + ' ' + collision_reports['address_sfx_intersecting']
collision_reports['street_name_intersecting'] = collision_reports['street_name_intersecting'].str.strip()
collision_reports.drop(columns = ['address_name_intersecting', 'address_sfx_intersecting'], inplace = True)

# Eliminating streets we don't have data for
collision_reports = collision_reports[collision_reports['street_name_primary'].isin(traffic_counts['street_name'])]
traffic_counts = traffic_counts[traffic_counts['street_name'].isin(collision_reports['street_name_primary'])]

# Save to csv
collision_reports.to_csv('collision_reports_processed.csv')
traffic_counts.to_csv('traffic_counts_processed.csv')

# Questions

# How to treat freeway ramps? Are collisions delegated to freeway, street of ramp name, or disregarded (treated as own entity?)

