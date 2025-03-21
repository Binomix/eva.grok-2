import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import io

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

# Функция обработки VCF/CSV
def process_vcf(file, model, scaler, X_train_columns):
    try:
        # Определяем разделитель в зависимости от расширения файла
        file_name = file.name.lower()
        if file_name.endswith('.vcf'):
            sep = '\t'  # VCF-файлы используют табуляцию
            file.seek(0)
            lines = file.read().decode('utf-8').splitlines()
            filtered_lines = [line for line in lines if not line.startswith('##')]
            filtered_file = io.StringIO('\n'.join(filtered_lines))
        else:
            sep = ','  # CSV-файлы используют запятые
            filtered_file = file

        # Читаем файл
        df = pd.read_csv(filtered_file, sep=sep)
        original_df = df.copy()
        df = df.drop('ID', axis=1)
        categorical_cols = ['CHROM', 'REF', 'ALT', 'FILTER', 'INFO']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)

        # Добавляем отсутствующие столбцы из X_train_columns
        missing_cols = [col for col in X_train_columns if col not in df_encoded.columns]
        for col in missing_cols:
            df_encoded[col] = 0

        # Убедимся, что все столбцы из X_train_columns присутствуют
        if not all(col in df_encoded.columns for col in X_train_columns):
            missing = [col for col in X_train_columns if col not in df_encoded.columns]
            raise KeyError(f"Не удалось добавить столбцы: {missing}")

        # Выравниваем столбцы
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
    except pd.errors.ParserError:
        st.error("Ошибка: Не удалось прочитать файл. Убедитесь, что это корректный CSV или VCF файл.")
        return None
    except KeyError as e:
        st.error(f"Ошибка: В файле отсутствует ожидаемый столбец: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Произошла ошибка при обработке файла: {str(e)}")
        return None

# Интерфейс Streamlit
st.title("Анализ ДНК-теста")
st.header("Загрузите ваш VCF-файл")
uploaded_file = st.file_uploader("Выберите файл", type=["csv", "vcf"])

if uploaded_file is not None:
    with st.spinner("Обработка файла..."):
        result = process_vcf(uploaded_file, model, scaler, X_train_columns)
    
    if result is not None:
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
