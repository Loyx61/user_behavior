import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns



#Анализ поведения пользователей в зависимости от устройства

try:
    df = pd.read_csv('user_behavior_dataset.csv')
    print("Данные успешно загружены.")
except FileNotFoundError:
    print("Ошибка: файл не найден. Проверьте путь и имя файла.")
    exit()

grouped_data = df.groupby(['Device Model', 'Operating System']).agg({
    'App Usage Time (min/day)': 'mean',
    'Screen On Time (hours/day)': 'mean',
    'Battery Drain (mAh/day)': 'mean'
}).reset_index()

print(grouped_data)

#Самые энергоэффективные устройства

df['Battery per Minute (mAh/min)'] = df['Battery Drain (mAh/day)'] / df['App Usage Time (min/day)']

energy_efficiency = df.groupby('Device Model')['Battery per Minute (mAh/min)'].mean().sort_values()

print("5 самых энергоэффективных устройств")
print(energy_efficiency.head())



#Классификация пользователей по уровню активности

low_threshold = df['App Usage Time (min/day)'].quantile(0.25)
high_threshold = df['App Usage Time (min/day)'].quantile(0.75)


df['Activity Level'] = pd.cut(
    df['App Usage Time (min/day)'],
    bins=[0, low_threshold, high_threshold, df['App Usage Time (min/day)'].max()],
    labels=['Low', 'Medium', 'High']
)


print(df['Activity Level'].value_counts())

#Уровень активности в зависимости от пола

print(pd.crosstab(df['Activity Level'], df['Gender']))




# Исследование влияния возраста на использование смартфона

cor_battery = df[["Battery Drain (mAh/day)", "App Usage Time (min/day)", "Screen On Time (hours/day)", "Number of Apps Installed"]].corr()
print(cor_battery)
#визуализация
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(
    cor_battery,
    annot=True,
    fmt=".2f",
    cmap='vlag',
    center=0,
    linewidths=0.5,
    annot_kws={"size": 10}
)
plt.title('Корреляция: Разряд батареи vs Активность пользователя', pad=20, fontsize=14)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()


def age_category(age):
    if age in range(12,21):
        return 'Подростки'
    elif age in range(20,36):
        return 'Молодежь'
    elif age in range(35,61):
        return 'Взрослые'
    else:
        return 'Другие'

df['Возрастная категория'] = df['Age'].apply(age_category)
print(df[['Age', 'Возрастная категория']])

age_stats = df.groupby('Возрастная категория').agg({
    'App Usage Time (min/day)': 'mean',
    'Number of Apps Installed': 'median',
    'Data Usage (MB/day)': 'mean'
}).sort_values('App Usage Time (min/day)', ascending=False)
print(age_stats)

corr_1 = df[['Age', 'Number of Apps Installed', 'Data Usage (MB/day)',"App Usage Time (min/day)"]].corr()
print(corr_1)
#визуализация
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_1,
    annot=True,
    cmap='coolwarm',
    center=0,
    annot_kws={'size': 12},
    linewidths=0.5
)
plt.title('Корреляция: Возраст vs Активность', pad=20)
plt.show()

for category in df['Возрастная категория'].unique():
    subset = df[df['Возрастная категория'] == category]
    print(f"\nКатегория: {category}")
    print(subset[['App Usage Time (min/day)', 'Data Usage (MB/day)']].corr())


#Анализ расхода батареи

battery_corr = df[['Battery Drain (mAh/day)',
                   'App Usage Time (min/day)',
                   'Screen On Time (hours/day)',
                   'Number of Apps Installed',
                   'Data Usage (MB/day)']].corr()

print("Корреляция с расходом батареи:")
print(battery_corr['Battery Drain (mAh/day)'].sort_values(ascending=False)[1:])

plt.figure(figsize=(10, 8))
sns.heatmap(battery_corr,
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5)
plt.title('Корреляция расхода батареи с параметрами использования', pad=20)
plt.tight_layout()
plt.show()

device_stats = df.groupby('Device Model')['Battery Drain (mAh/day)'].agg(['mean', 'std', 'count'])
device_stats = device_stats.sort_values('mean', ascending=False)

print("\nСредний расход батареи по устройствам:")
print(device_stats)


high_drain_threshold = df['Battery Drain (mAh/day)'].quantile(0.95)
high_drain_devices = df[df['Battery Drain (mAh/day)'] > high_drain_threshold]

print("\nУстройства с аномально высоким расходом (> 95% перцентиль):")
print(high_drain_devices['Device Model'].value_counts())