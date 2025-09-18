import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

country_map = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'HR': 'Croatia', 'CY': 'Cyprus',
    'CZ': 'Czechia', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France',
    'DE': 'Germany', 'EL': 'Greece', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta', 'NL': 'Netherlands',
    'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SK': 'Slovakia', 'SI': 'Slovenia',
    'ES': 'Spain', 'SE': 'Sweden',
    'IS': 'Iceland', 'NO': 'Norway', 'CH': 'Switzerland', 'TR': 'Turkey',
    'BA': 'Bosnia and Herzegovina','RS': 'Serbia',
    'EU27_2020': 'European Union'
}

origin_map = {
    'NAT': 'Native-born',
    'EU27_2020_FOR': 'EU-born Migrants',
    'NEU27_2020_FOR': 'Non-EU Migrants'
}

# Countries to exclude due to no data:
exclude_countries = ['MK', 'ME', 'UK']

def load_and_clean_wide_tsv(filepath):
    df_raw = pd.read_csv(filepath, sep='\t', header=0)
    split_cols = df_raw.iloc[:, 0].str.split(',', expand=True)
    split_cols.columns = ['freq', 'unit', 'sex', 'age', 'citizen', 'geo']
    df = pd.concat([split_cols, df_raw.iloc[:, 1:]], axis=1)
    df_long = df.melt(
        id_vars=['freq', 'unit', 'sex', 'age', 'citizen', 'geo'],
        var_name='time',
        value_name='value'
    )
    df_long['value'] = df_long['value'].astype(str)
    df_long['value'] = df_long['value'].replace(':', np.nan)
    df_long['value'] = df_long['value'].str.replace(r'\(.*?\)', '', regex=True)
    df_long['value'] = df_long['value'].str.replace(r'[^\d.,]', '', regex=True)
    df_long['value'] = df_long['value'].str.replace(',', '.', regex=False)
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long['time'] = df_long['time'].astype(int)
    df_filtered = df_long[
        (df_long['freq'] == 'A') &
        (df_long['unit'] == 'PC') &
        (df_long['age'] == 'Y20-64') &
        (~df_long['geo'].isin(exclude_countries))  # Remove unwanted countries here
    ].copy()
    # Map country and origin names
    df_filtered['country'] = df_filtered['geo'].map(country_map).fillna(df_filtered['geo'])
    df_filtered['origin'] = df_filtered['citizen'].map(origin_map).fillna(df_filtered['citizen'])
    return df_filtered

df_emp = load_and_clean_wide_tsv('estat_lfsa_ergan_filtered.tsv')
df_unemp = load_and_clean_wide_tsv('estat_lfsa_urgan_filtered.tsv')

df_emp_2023 = df_emp[(df_emp['time'] == 2023) & (df_emp['sex'] == 'T')]
df_unemp_2023 = df_unemp[(df_unemp['time'] == 2023) & (df_unemp['sex'] == 'T')]

def pivot_citizenship(df):
    pivot = df.pivot(index='country', columns='origin', values='value')
    return pivot

emp_pivot = pivot_citizenship(df_emp_2023)
unemp_pivot = pivot_citizenship(df_unemp_2023)

# --- 1. Employment barplot with markers ---

# Sort by native-born employment rate (descending)
emp_sorted = emp_pivot.sort_values(by='Native-born', ascending=False)

plt.figure(figsize=(14, 7))
bars = plt.bar(emp_sorted.index, emp_sorted['Native-born'], color='skyblue', label='Native-born')
plt.scatter(emp_sorted.index, emp_sorted['EU-born Migrants'], color='orange', label='EU-born Migrants', zorder=5)
plt.scatter(emp_sorted.index, emp_sorted['Non-EU Migrants'], color='green', label='Non-EU Migrants', zorder=5)
plt.xticks(rotation=90)
plt.ylabel('Employment Rate (%)')
plt.title('Employment Rate by Citizenship, Persons Aged 20–64 (2023)\n(Sorted by Native-born Rate)')
plt.legend()
plt.tight_layout()
plt.show()


# --- 2. Unemployment barplot with markers ---

# Sort by native-born unemployment rate (descending)
unemp_sorted = unemp_pivot.sort_values(by='Native-born', ascending=False)

plt.figure(figsize=(14, 7))
bars = plt.bar(unemp_sorted.index, unemp_sorted['Native-born'], color='salmon', label='Native-born')
plt.scatter(unemp_sorted.index, unemp_sorted['EU-born Migrants'], color='blue', label='EU-born Migrants', zorder=5)
plt.scatter(unemp_sorted.index, unemp_sorted['Non-EU Migrants'], color='darkred', label='Non-EU Migrants', zorder=5)
plt.xticks(rotation=90)
plt.ylabel('Unemployment Rate (%)')
plt.title('Unemployment Rate by Citizenship, Persons Aged 20–64 (2023)\n(Sorted by Native-born Rate)')
plt.legend()
plt.tight_layout()
plt.show()



# --- 3: Employment rate over years for EU, Czechia, Germany ---

countries_to_plot = ['European Union', 'Czechia', 'Germany']

plt.figure(figsize=(12,8))
for country in countries_to_plot:
    sub = df_emp[(df_emp['country'] == country) & (df_emp['sex'] == 'T')]
    sub_pivot = sub.pivot(index='time', columns='origin', values='value')
    plt.plot(sub_pivot.index, sub_pivot['Native-born'], label=f'{country} Native-born', linewidth=2)
    plt.plot(sub_pivot.index, sub_pivot['EU-born Migrants'], label=f'{country} EU-born Migrants', linestyle='--')

plt.xlabel('Year')
plt.ylabel('Employment Rate (%)')
plt.title('Employment Rate Over Time by Citizenship and Country')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 4: Heatmap of unemployment rate in 2023 by country and citizenship ---

plt.figure(figsize=(10,12))
sns.heatmap(unemp_pivot, annot=True, fmt=".1f", cmap='Reds', cbar_kws={'label': 'Unemployment Rate (%)'})
plt.title('Unemployment Rate by Country and Citizenship (2023)')
plt.ylabel('Country')
plt.xlabel('Citizenship')
plt.tight_layout()
plt.show()
