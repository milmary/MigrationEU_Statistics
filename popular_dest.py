import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import re

# --------------------------
# 1. Load and Preprocess Data
# --------------------------

# Country code to name map
country_map = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'HR': 'Croatia', 'CY': 'Cyprus',
    'CZ': 'Czechia', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France',
    'DE': 'Germany', 'EL': 'Greece', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta', 'NL': 'Netherlands',
    'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SK': 'Slovakia', 'SI': 'Slovenia',
    'ES': 'Spain', 'SE': 'Sweden', 'IS': 'Iceland', 'NO': 'Norway', 'CH': 'Switzerland',
    'TR': 'Turkey', 'UK': 'United Kingdom', 'ME': 'Montenegro', 'MK': 'North Macedonia',
    'BA': 'Bosnia and Herzegovina', 'RS': 'Serbia', 'EU27_2020': 'European Union'
}

df_raw = pd.read_csv("estat_migr_resfirst_filtered.tsv", sep='\t', header=None)
df_split = df_raw[0].str.split(',', expand=True)
df_split.columns = ['freq', 'reason', 'citizen', 'duration', 'unit', 'geo']
df_split['value'] = df_raw[1]

# Clean up
# Clean and convert value column
df_split['value'] = (
    df_split['value']
    .astype(str)
    .str.replace(r'\(.*?\)', '', regex=True)
    .str.strip()
    .str.replace(r'[^\d.]', '', regex=True)
)
df_split['value'] = pd.to_numeric(df_split['value'], errors='coerce')

# Add country column before filtering
df_split['geo'] = df_split['geo'].str.strip()
df_split['country'] = df_split['geo'].map(country_map)

# Drop invalid rows
df_split = df_split.dropna(subset=['value', 'country'])
df = df_split[
    (df_split['duration'] == 'TOTAL') &
    (df_split['freq'] == 'A') &
    (df_split['unit'] == 'PER')
]

# Group and pivot data
df_grouped = df.groupby(['country', 'reason'])['value'].sum().reset_index()
df_pivot = df_grouped.pivot(index='country', columns='reason', values='value').fillna(0)

# --------------------------
# 2. Prepare Map of Europe
# --------------------------

# Load shapefile manually
world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

# Filter Europe
europe = world[world['CONTINENT'] == 'Europe'].copy()
europe = europe.replace({'NAME': {
    'Czech Republic': 'Czechia',
    'Bosnia and Herz.': 'Bosnia and Herzegovina'
}})

def plot_map(reason, title, cmap='Blues'):
    # Prepare data
    data = df_pivot[[reason]].reset_index()
    merged = europe.merge(data, how='left', left_on='NAME', right_on='country')

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    merged.plot(
        column=reason,
        cmap=cmap,
        scheme='quantiles',
        k=6,
        legend=True,
        edgecolor='black',
        linewidth=0.5,
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "black",
            "hatch": "///",
            "label": "No data"
        }
    )

    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    # Set x and y limits to focus on continental Europe
    ax.set_xlim(-25, 45)
    ax.set_ylim(34, 72)
    
    plt.tight_layout()
    plt.show()

# --------------------------
# 3. Visualizations
# --------------------------

# 1. Map: Educational Residence Permits
plot_map('EDUC', 'Educational Residence Permits in Europe (2023)', cmap='Blues')

# 2. Map: Employment Residence Permits
plot_map('EMP', 'Employment Residence Permits in Europe (2023)', cmap='Oranges')

# 3. Horizontal barplot: Total Permits
if 'TOTAL' in df_pivot.columns:
    total_sorted = df_pivot['TOTAL'].sort_values(ascending=True)
    plt.figure(figsize=(10, 12))
    sns.barplot(x=total_sorted.values, y=total_sorted.index, palette='viridis')
    plt.xlabel("Number of Total Residence Permits")
    plt.title("Total Residence Permits by Country (2023)")
    plt.tight_layout()
    plt.show()

# 4. Stacked Bar Chart: Proportion of permit reasons per country
available_reasons = [col for col in ['FAM', 'EDUC', 'EMP', 'OTH'] if col in df_pivot.columns]
df_stack = df_pivot[available_reasons]
df_stack = df_stack.div(df_stack.sum(axis=1), axis=0)

# Sort by EDUC proportion (highest to lowest) if EDUC exists
if 'EDUC' in df_stack.columns:
    df_sorted_educ = df_stack.sort_values(by='EDUC', ascending=False)

    df_sorted_educ.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='Set2')
    plt.title('Proportion of Permit Reasons per Country (Sorted by EDUC, 2023)')
    plt.ylabel('Proportion')
    plt.xlabel('Country')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Sort by EMP proportion (highest to lowest) if EMP exists
if 'EMP' in df_stack.columns:
    df_sorted_emp = df_stack.sort_values(by='EMP', ascending=False)

    df_sorted_emp.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='Set2')
    plt.title('Proportion of Permit Reasons per Country (Sorted by EMP, 2023)')
    plt.ylabel('Proportion')
    plt.xlabel('Country')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
