# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

# Matplotlib visualization
import matplotlib.pyplot as plt

# Set default font size
plt.rcParams['font.size'] = 24

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Splitting data into training and testing
from sklearn.model_selection import train_test_split
i=1
# Read in data into a dataframe 
data = pd.read_csv('data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')

# Display top of dataframe
data.head()
data.info()

# Replace all occurrences of Not Available with numpy not a number
data = data.replace({'Not Available': np.nan})

# Iterate through the columns
for col in list(data.columns):
    # Select columns that should be numeric
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 
        col or 'therms' in col or 'gal' in col or 'Score' in col):
        # Convert the data type to float
        data[col] = data[col].astype(float)
data.describe()

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values_table(data)
missing_df = missing_values_table(data);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))

data = data.drop(columns = list(missing_columns))


# Rename the score 
data = data.rename(columns = {'ENERGY STAR Score': 'score'})


# Histogram of the Energy Star Score
# plt.style.use('fivethirtyeight')
# plt.hist(data['score'].dropna(), bins = 100, edgecolor = 'k');
# plt.xlabel('Score'); plt.ylabel('Number of Buildings'); 
# plt.title('Energy Star Score Distribution');
##plt.show()
plt.savefig(f'Res{i}.png')
i=i+1

first_quartile = data['Site EUI (kBtu/ft²)'].describe()['25%']
third_quartile = data['Site EUI (kBtu/ft²)'].describe()['75%']

# Interquartile range
iqr = third_quartile - first_quartile

# Remove outliers
data = data[(data['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * iqr)) &
            (data['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * iqr))]

# plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');
# plt.xlabel('Site EUI'); 
# plt.ylabel('Count'); plt.title('Site EUI Distribution');
# ##plt.show()
# plt.savefig(f'Res{i}.png')
i=i+1



types = data.dropna(subset=['score'])
types = types['Largest Property Use Type'].value_counts()
types = list(types[types.values > 100].index)

for b_type in types:
    # Select the building type
    subset = data[data['Largest Property Use Type'] == b_type]
    scores_array = subset['score'].dropna().to_numpy().reshape(-1, 1)
    # Density plot of Energy Star scores
    sns.kdeplot(scores_array,
               label = b_type, fill=True)
    
# label the plot
plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Energy Star Scores by Building Type', size = 28);
#plt.show()
plt.savefig(f'Res{i}.png')
i=i+1



boroughs = data.dropna(subset=['score'])
boroughs = boroughs['Borough'].value_counts()
boroughs = list(boroughs[boroughs.values > 100].index)



figsize(12, 10)

# Plot each borough distribution of scores
for borough in boroughs:
    # Select the building type
    subset = data[data['Borough'] == borough]
    
    scores_array = subset['score'].dropna().to_numpy().reshape(-1, 1)
    # Density plot of Energy Star scores
    sns.kdeplot(scores_array,
               label = borough, fill=True)
    
# label the plot
plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Energy Star Scores by Borough', size = 28);
#plt.show()
plt.savefig(f'Res{i}.png')
i=i+1
# Find all correlations and sort 
numerical_data = data.select_dtypes(include=[np.number])
correlations_data = numerical_data.corr()['score'].sort_values()


# Print the most negative correlations
print(correlations_data.head(15), '\n')

# Print the most positive correlations
print(correlations_data.tail(15))

# Установка размера фигуры
plt.figure(figsize=(12, 10))

# Выбор числовых столбцов
numeric_subset = data.select_dtypes('number')

# Создание столбцов с квадратным корнем и логарифмом числовых столбцов
for col in numeric_subset.columns:
    # Пропуск столбца Energy Star Score
    if col == 'score':
        continue
    else:
        # Обработка отрицательных значений для sqrt и log
        numeric_subset['sqrt_' + col] = np.sqrt(np.maximum(numeric_subset[col], 0))
        numeric_subset['log_' + col] = np.log1p(numeric_subset[col])

# Выбор категориальных столбцов
categorical_subset = data[['Borough', 'Largest Property Use Type']]

# Применение one-hot кодирования
categorical_subset = pd.get_dummies(categorical_subset)

# Объединение двух датафреймов с помощью concat
# Используйте axis = 1 для объединения столбцов
features = pd.concat([numeric_subset, categorical_subset], axis=1)

# Удаление зданий без оценки Energy Star Score
features = features.dropna(subset=['score'])

# Поиск корреляций с оценкой
correlations = features.corr()['score'].dropna().sort_values()

# Вывод корреляций
print(correlations)

# Извлечение типов зданий
features['Largest Property Use Type'] = data.dropna(subset=['score'])['Largest Property Use Type']

# Ограничение до типов зданий с более чем 100 наблюдениями (из предыдущего кода)
features = features[features['Largest Property Use Type'].isin(types)]

# Использование seaborn для создания точечной диаграммы Score vs Log Source EUI
# sns.lmplot(x='Site EUI (kBtu/ft²)', y='score', 
#            hue='Largest Property Use Type', data=features,
#            scatter_kws={'alpha': 0.8, 's': 60}, fit_reg=False,
#            height=10, aspect=1.2)

# Подписи к графику
# plt.xlabel("Site EUI", size=28)
# plt.ylabel('Energy Star Score', size=28)
# plt.title('Energy Star Score vs Site EUI', size=36)
# #plt.show()
# plt.savefig(f'Res{i}.png')
i=i+1



plot_data = features[['score', 'Site EUI (kBtu/ft²)', 
                      'Weather Normalized Source EUI (kBtu/ft²)', 
                      'log_Total GHG Emissions (Metric Tons CO2e)']]

# Replace the inf with nan
plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})

# Rename columns 
plot_data = plot_data.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI', 
                                        'Weather Normalized Source EUI (kBtu/ft²)': 'Weather Norm EUI',
                                        'log_Total GHG Emissions (Metric Tons CO2e)': 'log GHG Emissions'})

# Drop na values
plot_data = plot_data.dropna()

def corr_func(x, y, **kwargs):
    if not np.isnan(x).any() and not np.isnan(y).any():
        r = np.corrcoef(x, y)[0][1]
        ax = plt.gca()
        ax.annotate(f"r = {r:.2f}", xy=(.2, .8), xycoords=ax.transAxes, size = 20)

# # Подготовка данных для визуализации
# plot_data = features[['score', 'Site EUI (kBtu/ft²)', 
#                       'Weather Normalized Source EUI (kBtu/ft²)', 
#                       'log_Total GHG Emissions (Metric Tons CO2e)']]
# plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})
# plot_data = plot_data.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI', 
#                                         'Weather Normalized Source EUI (kBtu/ft²)': 'Weather Norm EUI',
#                                         'log_Total GHG Emissions (Metric Tons CO2e)': 'log GHG Emissions'})
# plot_data = plot_data.dropna()

# # Создание объекта PairGrid
# grid = sns.PairGrid(data = plot_data, height = 3)

# # Верхняя часть - точечная диаграмма
# grid.map_upper(plt.scatter, color = 'red', alpha = 0.6)

# # Диагональ - гистограмма
# grid.map_diag(plt.hist, color = 'red', edgecolor = 'black')

# # Нижняя часть - корреляция и плотность распределения
# grid.map_lower(corr_func)
# grid.map_lower(sns.kdeplot, cmap = plt.cm.Reds)

# # Заголовок для всего графика
# plt.suptitle('Pairs Plot of Energy Data', size = 36, y = 1.02)

# # Показать график
# #plt.show()
# plt.savefig(f'Res{i}.png')
i=i+1


# Копирование исходных данных
features = data.copy()

# Выбор числовых столбцов
numeric_subset = data.select_dtypes('number')

# Создание столбцов с логарифмом числовых столбцов
for col in numeric_subset.columns:
    # Пропуск столбца Energy Star Score
    if col == 'score':
        continue
    else:
        # Проверка на отрицательные значения и нули
        if (numeric_subset[col] <= 0).any():
            numeric_subset['log_' + col] = numeric_subset[col].apply(lambda x: np.log(x) if x > 0 else np.nan)
        else:
            numeric_subset['log_' + col] = np.log(numeric_subset[col])

# Выбор категориальных столбцов
categorical_subset = data[['Borough', 'Largest Property Use Type']]

# Применение one-hot кодирования
categorical_subset = pd.get_dummies(categorical_subset)

# Объединение двух датафреймов с помощью concat
# Используйте axis = 1 для объединения столбцов
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

# Вывод размера датафрейма
print(features.shape)

# Подготовка данных для графика
plot_data = data[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna()

# Преобразование данных в массивы NumPy
site_eui = plot_data['Site EUI (kBtu/ft²)'].values
weather_norm_eui = plot_data['Weather Normalized Site EUI (kBtu/ft²)'].values

# Расчёт коэффициента корреляции
correlation = np.corrcoef(site_eui, weather_norm_eui)[0][1]

# Создание графика
plt.plot(site_eui, weather_norm_eui, 'bo')
plt.xlabel('Site EUI')
plt.ylabel('Weather Norm EUI')
plt.title(f'Weather Norm EUI vs Site EUI, R = {correlation:.4f}')
plt.grid(True) # Добавление сетки

# Отображение графика
#plt.show()
plt.savefig(f'Res{i}.png')
i=i+1