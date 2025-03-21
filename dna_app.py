import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Загрузка модели, scaler и columns
model = load_model('dna_model.keras')
scaler = joblib.load('scaler.pkl')
X_train_columns = joblib.load('columns.pkl')

# Словарь болезней
disease_map = {
    'F508del': 'Муковисцидоз',
    'exon7del': 'Спинальная мышечная атрофия (СМА)',
    'R408W': 'Фенилкетонурия',
    'HbS': 'Серповидноклеточная анемия',
    '1278insTATC': 'Болезнь Тея-Сакса',
    'CGGexp': 'Синдром ломкой Х-хромосомы',
    'CAGexp': 'Хорея Хантингтона'
}

# Функция обработки VCF
def process_vcf(file, model, scaler, X_train_columns):
    # Определяем разделитель в зависимости от расширения файла
    file_name = file.name.lower()
    if file_name.endswith('.vcf'):
        sep = '\t'  # VCF-файлы используют табуляцию
        skiprows = lambda x: x.startswith('##')  # Пропускаем строки с метаданными
    else:
        sep = ','  # CSV-файлы используют запятые
        skiprows = None

    # Читаем файл
    df = pd.read_csv(file, sep=sep, skiprows=skiprows)
    original_df = df.copy()
    df = df.drop('ID', axis=1)
    categorical_cols = ['CHROM', 'REF', 'ALT', 'FILTER', 'INFO']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    for col in X_train_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[X_train_columns]
    numeric_cols = ['POS', 'QUAL']
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
    predictions = model.predict(df_encoded)
    original_df['PREDICTED_PATHOGENIC'] = (predictions > 0.5).astype(int)
    original_df['DISEASE'] = original_df['INFO'].apply(
        lambda x: next((disease_map[v.split('VARIANT=')[1]] for v in x.split(';') if 'VARIANT=' in v), 'Неизвестно')
    )
    result = original_df[original_df['PREDICTED_PATHOGENIC'] == 1][['CHROM', 'POS', 'REF', 'ALT', 'INFO', 'DISEASE']]
    return result

# Интерфейс Streamlit
st.title("Анализ ДНК-теста")
st.header("Загрузите ваш VCF-файл")
uploaded_file = st.file_uploader("Выберите файл", type=["csv", "vcf"])

if uploaded_file is not None:
    with st.spinner("Обработка файла..."):
        result = process_vcf(uploaded_file, model, scaler, X_train_columns)
        dna_passport = result['DISEASE'].value_counts().reset_index()
        dna_passport.columns = ['Заболевание', 'Количество мутаций']
    
    st.header("Результаты анализа")
    tab1, tab2 = st.tabs(["Полные результаты", "ДНК-паспорт"])
    
    with tab1:
        st.subheader("Обнаруженные патогенные мутации")
        st.dataframe(result)
    
    with tab2:
        st.subheader("Ваш ДНК-паспорт")
        st.table(dna_passport)
