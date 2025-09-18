import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style="whitegrid")

# --- 1. Load & Clean Data ---
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df = df.rename(columns={df.columns[0]: 'meta'})
    meta_split = df['meta'].str.split(',', expand=True)
    meta_split.columns = ['freq', 'unit', 'sex', 'isced11', 'c_birth', 'age', 'geo']
    df = pd.concat([meta_split, df.drop(columns=['meta'])], axis=1)
    df = df.melt(id_vars=['freq', 'unit', 'sex', 'isced11', 'c_birth', 'age', 'geo'],
                 var_name='time', value_name='value')
    df['time'] = df['time'].str.strip()
    df['value'] = df['value'].str.extract(r'([\d.,]+)')
    df['value'] = df['value'].str.replace(',', '.', regex=False).astype(float)
    return df

# --- 2. Label Mappings ---
origin_map = {
    'NAT': 'Native-born',
    'EU27_2020_FOR': 'EU-born Migrants',
    'NEU27_2020_FOR': 'Non-EU Migrants'
}
edu_map = {
    'ED0-2': 'Low',
    'ED3_4': 'Medium',
    'ED5-8': 'High'
}

# --- Country Code to Name Mapping ---
country_map = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'HR': 'Croatia', 'CY': 'Cyprus',
    'CZ': 'Czechia', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France',
    'DE': 'Germany', 'EL': 'Greece', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta', 'NL': 'Netherlands',
    'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SK': 'Slovakia', 'SI': 'Slovenia',
    'ES': 'Spain', 'SE': 'Sweden',
    'IS': 'Iceland', 'NO': 'Norway', 'CH': 'Switzerland', 'TR': 'Turkey', 'UK': 'United Kingdom',
    'ME': 'Montenegro', 'MK': 'North Macedonia', 'BA': 'Bosnia and Herzegovina','RS': 'Serbia',
    'EU27_2020': 'European Union'
}

# --- Helper Function ---
def map_country_names(df, column='geo'):
    df = df.copy()
    df[column] = df[column].map(country_map).fillna(df[column])
    return df


# --- 3. Utility ---
def plot_and_save(fig, pdf):
    pdf.savefig(fig)
    plt.close(fig)

# --- 4. Plotting Functions --
def plot_q1(df, pdf):
    year, countries = '2023', ['IE', 'FR', 'LV', 'CZ', 'GR', 'TR', 'PL']
    f = df[(df['time'] == year) & (df['sex'] == 'T') & (df['age'] == 'Y25-64') &
           (df['c_birth'].isin(origin_map)) & (df['isced11'].isin(edu_map)) & (df['geo'].isin(countries))].copy()
    f['origin'] = f['c_birth'].map(origin_map)
    f['education'] = f['isced11'].map(edu_map)
    f = map_country_names(f, 'geo')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=f, x='geo', y='value', hue='education', ax=ax, errorbar=None)
    ax.set_title(f'Education Level by Origin ({year})')
    ax.set_xlabel('Country')
    ax.set_ylabel('Percentage')
    ax.legend(title='Education Level')
    plot_and_save(fig, pdf)


def plot_q2(df, pdf):
    f = df[(df['geo'] == 'EU27_2020') & (df['c_birth'] == 'NEU27_2020_FOR') &
           (df['sex'] == 'T') & (df['age'] == 'Y15-64') & (df['isced11'].isin(edu_map))].copy()
    f['education'] = f['isced11'].map(edu_map)
    f['time'] = f['time'].astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=f, x='time', y='value', hue='education', marker='o', ax=ax)
    ax.set_title('Trend: Education of Non-EU Migrants in EU (2015–2024)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage')
    ax.legend(title='Education Level')
    plot_and_save(fig, pdf)


def plot_q3(df, pdf):
    year = '2023'
    excluded = ['BA', 'ME', 'MK', 'UK', 'BG']
    f = df[(df['isced11'] == 'ED5-8') & (df['sex'] == 'T') & (df['age'] == 'Y25-54') &
           (df['time'] == year) & (df['c_birth'].isin(['EU27_2020_FOR', 'NEU27_2020_FOR'])) & 
           (~df['geo'].isin(excluded))]
    grouped = f.groupby('geo')['value'].mean().sort_values(ascending=False).reset_index()
    grouped = map_country_names(grouped, 'geo')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=grouped, x='value', y='geo', palette='coolwarm', ax=ax, legend=False)
    ax.set_title(f'Highly Educated Migrants by Country ({year})')
    ax.set_xlabel('Percentage with Tertiary Education')
    ax.set_ylabel('Country')
    plot_and_save(fig, pdf)



def plot_q4(df, pdf):
    year, countries = '2023', ['IE', 'CZ', 'FR', 'IT']
    f = df[(df['isced11'] == 'ED5-8') & (df['sex'] == 'T') & (df['age'] == 'Y25-54') &
           (df['time'] == year) & (df['c_birth'].isin(origin_map)) & (df['geo'].isin(countries))].copy()
    f['origin'] = f['c_birth'].map(origin_map)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=f, x='geo', y='value', hue='origin', palette='Set2', ax=ax)
    ax.set_title(f'Tertiary Education: Migrants vs Natives ({year})')
    ax.set_xlabel('Country')
    ax.set_ylabel('Percentage')
    ax.legend(title='Population Group')
    plot_and_save(fig, pdf)


def plot_q5(df, pdf):
    f = df[(df['time'] == '2023') & (df['geo'] == 'EU27_2020') &
           (df['age'] == 'Y25-34') & (df['isced11'] == 'ED5-8') &
           (df['c_birth'].isin(origin_map))].copy()
    f['origin'] = f['c_birth'].map(origin_map)
    f['sex'] = f['sex'].map({'F': 'Female', 'M': 'Male', 'T': 'Total'})
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=f, x='sex', y='value', hue='origin')
    plt.title('Tertiary Education (25–34) by Sex and Origin (EU, 2023)')
    plt.ylabel('Percentage')
    plt.xlabel(None)
    plt.legend(title='Population Group')
    plot_and_save(fig, pdf)


def plot_q6(df, pdf):
    order = ['Low', 'Medium', 'High']
    f = df[(df['time'] == '2023') & (df['geo'] == 'EU27_2020') &
           (df['age'] == 'Y25-54') & (df['c_birth'].isin(origin_map)) &
           (df['isced11'].isin(edu_map)) & (df['sex'] == 'T')].copy()
    f['origin'] = f['c_birth'].map(origin_map)
    f['education'] = f['isced11'].map(edu_map)
    f['education'] = pd.Categorical(f['education'], categories=order, ordered=True)
    pivot = f.pivot_table(index='education', columns='origin', values='value').fillna(0)
    ax = pivot.T.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Education Level by Origin (25–54, EU, 2023)')
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.xticks(rotation=0)
    plt.legend(title='Education Level')
    plt.tight_layout()
    fig = ax.get_figure()
    plot_and_save(fig, pdf)


def plot_q7(df, pdf):
    order = ['Low', 'Medium', 'High']
    f = df[(df['time'] == '2023') & (df['geo'] == 'EU27_2020') &
           (df['age'] == 'Y25-54') & (df['c_birth'].isin(origin_map)) &
           (df['isced11'].isin(edu_map)) & (df['sex'] != 'T')].copy()

    f['origin'] = f['c_birth'].map(origin_map)
    f['education'] = f['isced11'].map(edu_map)
    f['education'] = pd.Categorical(f['education'], categories=order, ordered=True)
    f['sex'] = f['sex'].map({'F': 'Female', 'M': 'Male'})

    origins = f['origin'].unique()
    bar_width = 0.8
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d73027', '#fee08b', '#1a9850']
    xticks = []
    xticklabels = []

    for i, origin in enumerate(origins):
        for j, sex in enumerate(['Female', 'Male']):
            bar_index = i * 3 + j
            group = f[(f['origin'] == origin) & (f['sex'] == sex)]
            heights = group.sort_values('education')['value'].values
            bottom = 0
            for k, h in enumerate(heights):
                ax.bar(bar_index, h, bottom=bottom, width=bar_width, color=colors[k])
                bottom += h
            xticks.append(bar_index)
            xticklabels.append(sex)
        group_heights = f[f['origin'] == origin].groupby('sex')['value'].sum().values
        max_height = max(group_heights)
        y_offset = max_height * 0.05 
        mid = i * 3 + 0.5
        ax.text(mid, max_height + y_offset, origin, ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, max_height * 1.2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_title('Education by Sex and Origin (25–54, EU, 2023)')
    ax.set_ylabel('Percentage')
    ax.legend(order, title='Education Level')
    plt.tight_layout()
    plot_and_save(fig, pdf)


def plot_q8(df, pdf):
    order = ['Low', 'Medium', 'High']
    f = df[(df['time'] == '2023') & (df['geo'] == 'EU27_2020') &
           (df['age'].isin(['Y25-54', 'Y55-74'])) & (df['c_birth'].isin(origin_map)) &
           (df['isced11'].isin(edu_map)) & (df['sex'] == 'T')].copy()

    f['origin'] = f['c_birth'].map(origin_map)
    f['education'] = f['isced11'].map(edu_map)
    f['education'] = pd.Categorical(f['education'], categories=order, ordered=True)
    f['age'] = f['age'].map({'Y25-54': '25–54', 'Y55-74': '55–74'})

    origins = f['origin'].unique()
    bar_width = 0.8
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d73027', '#fee08b', '#1a9850']
    xticks = []
    xticklabels = []

    for i, origin in enumerate(origins):
        for j, age in enumerate(['25–54', '55–74']):
            bar_index = i * 3 + j
            group = f[(f['origin'] == origin) & (f['age'] == age)]
            heights = group.sort_values('education')['value'].values
            bottom = 0
            for k, h in enumerate(heights):
                ax.bar(bar_index, h, bottom=bottom, width=bar_width, color=colors[k])
                bottom += h
            xticks.append(bar_index)
            xticklabels.append(age)
        group_heights = f[f['origin'] == origin].groupby('age')['value'].sum().values
        max_height = max(group_heights)
        y_offset = max_height * 0.05
        mid = i * 3 + 0.5
        ax.text(mid, max_height + y_offset, origin, ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, max_height * 1.2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_title('Education by Age and Origin (EU, 2023)')
    ax.set_ylabel('Percentage')
    ax.legend(order, title='Education Level')
    plt.tight_layout()
    plot_and_save(fig, pdf)


def plot_q9(df, pdf):
    f = df[(df['age'] == 'Y25-34') & (df['isced11'] == 'ED5-8') &
           (df['sex'] == 'T') & (df['c_birth'].isin(origin_map))].copy()

    f['origin'] = f['c_birth'].map(origin_map)
    data_2015 = f[f['time'] == '2015'].groupby('origin')['value'].mean()
    data_2023 = f[f['time'] == '2023'].groupby('origin')['value'].mean()
    origins = list(data_2015.index)
    
    x_2015 = np.arange(len(origins))
    x_2023 = x_2015 + len(origins) + 1 

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Native-born': '#1f77b4', 'EU-born Migrants': '#ff7f0e', 'Non-EU Migrants': '#2ca02c'}
    for i, origin in enumerate(origins):
        ax.bar(x_2015[i], data_2015[origin], color=colors[origin])
        ax.bar(x_2023[i], data_2023[origin], color=colors[origin])

    # X-axis ticks and labels
    all_positions = list(x_2015) + list(x_2023)
    all_labels = [''] * len(all_positions)
    ax.set_xticks([(x_2015[0] + x_2015[-1]) / 2, (x_2023[0] + x_2023[-1]) / 2])
    ax.set_xticklabels(['2015', '2023'])

    ax.set_title('Tertiary Education (25–34) by Origin, 2015 vs 2023')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('')
    ax.legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in colors.values()],
              labels=colors.keys(), title='Population Group')
    plt.tight_layout()
    plot_and_save(fig, pdf)


def plot_q10(df, pdf):
    excluded = ['ME', 'MK', 'UK', 'BA']
    native = df[(df['time'] == '2023') & (df['age'] == 'Y25-34') &
                (df['isced11'] == 'ED5-8') & (df['sex'] == 'T') &
                (df['c_birth'] == 'NAT') & (~df['geo'].isin(excluded))].copy()
    migrant = df[(df['time'] == '2023') & (df['age'] == 'Y25-34') &
                 (df['isced11'] == 'ED5-8') & (df['sex'] == 'T') &
                 (df['c_birth'].isin(['EU27_2020_FOR', 'NEU27_2020_FOR'])) &
                 (~df['geo'].isin(excluded))].copy()
    
    native = map_country_names(native, 'geo')
    migrant = map_country_names(migrant, 'geo')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=native, x='geo', y='value', color='skyblue', ax=ax)
    
    for c_birth, marker in zip(['EU27_2020_FOR', 'NEU27_2020_FOR'], ['s', 'o']):
        group = migrant[migrant['c_birth'] == c_birth]
        ax.scatter(group['geo'], group['value'], label=origin_map[c_birth], marker=marker, s=100)
        for i, row in group.iterrows():
            native_val = native[native['geo'] == row['geo']]['value']
            if not native_val.empty:
                ax.plot([row['geo'], row['geo']], [native_val.values[0], row['value']], color='gray', linestyle='--')

    ax.set_title('Tertiary Education (25–34) by Native and Migrant Groups (2023)')
    ax.set_xlabel('Country')
    ax.set_ylabel('Percentage')
    ax.legend(title='Migrant Origin')
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plot_and_save(fig, pdf)


# --- 5. Main ---
if __name__ == '__main__':
    file_path = 'estat_edat_lfs_9912_filtered1.tsv'
    df = load_and_clean_data(file_path)
    with PdfPages('education_analysis.pdf') as pdf:
        plot_q1(df, pdf)
        plot_q2(df, pdf)
        plot_q3(df, pdf)
        plot_q4(df, pdf)
        plot_q5(df, pdf)
        plot_q6(df, pdf)
        plot_q7(df, pdf)
        plot_q8(df, pdf)
        plot_q9(df, pdf)
        plot_q10(df, pdf)
    print("All plots saved to education_analysis.pdf")