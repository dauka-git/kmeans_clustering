import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Bank Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

kmeans_model, scaler = load_model_and_scaler()


# Функция для описания кластеров
def get_cluster_description(cluster_num):
    """Возвращает описание кластера на основе его номера"""
    descriptions = {
        0: {
            "name": "💰 Консервативные клиенты",
            "description": """
            **Основные характеристики:**
            - Низкие расходы и баланс
            - Редко используют кредитный лимит
            - Минимальные транзакции
            
            **Рекомендации для банка:**
            - Предложить программы лояльности
            - Образовательные материалы о финансовых продуктах
            - Стимулировать использование карты
            """
        },
        1: {
            "name": "🚀 Активные транжиры", 
            "description": """
            **Основные характеристики:**
            - Высокие расходы и частые покупки
            - Большой кредитный лимит
            - Активное использование рассрочки
            
            **Рекомендации для банка:**
            - Премиальные программы и кэшбэк
            - Увеличение кредитного лимита
            - Персональные предложения от партнеров
            """
        },
        2: {
            "name": "📊 Сбалансированные пользователи",
            "description": """
            **Основные характеристики:**
            - Умеренные расходы
            - Стабильный баланс
            - Периодическое использование кредита
            
            **Рекомендации для банка:**
            - Сбалансированные пакеты услуг
            - Программы сбережений
            - Стандартные предложения по кредитам
            """
        }
        
    }
    
    return descriptions.get(cluster_num, {
        "name": "Неизвестный сегмент",
        "description": "Описание для этого сегмента пока не добавлено."
    })

def main():
    
    st.title("Сегментация клиентов банка")
    
    st.sidebar.header("Ввод данных клиента")
    st.sidebar.markdown("Введите характеристики клиента для классификации:")
    
    
    if kmeans_model is None or scaler is None:
        st.stop()
    
    with st.sidebar.form("customer_data_form"):
        st.subheader("Финансовые показатели")
        
        balance = st.number_input("Баланс (Balance)", min_value=0.0, value=1000.0, step=100.0)
        balance_frequency = st.slider("Частота пополнения баланса (Balance Frequency)", 0.0, 1.0, 0.5, 0.01)
        purchases = st.number_input("Сумма покупок (Purchases)", min_value=0.0, value=500.0, step=50.0)
        oneoff_purchases = st.number_input("Разовые покупки (OneOff Purchases)", min_value=0.0, value=200.0, step=50.0)
        installments_purchases = st.number_input("Покупки в рассрочку (Installments Purchases)", min_value=0.0, value=150.0, step=50.0)
        cash_advance = st.number_input("Авансы наличными (Cash Advance)", min_value=0.0, value=0.0, step=50.0)
        
        st.subheader("Частота операций")
        purchases_frequency = st.slider("Частота покупок (Purchases Frequency)", 0.0, 1.0, 0.5, 0.01)
        oneoff_purchases_frequency = st.slider("Частота разовых покупок (OneOff Purchases Frequency)", 0.0, 1.0, 0.3, 0.01)
        purchases_installments_frequency = st.slider("Частота покупок в рассрочку (Purchases Installments Frequency)", 0.0, 1.0, 0.3, 0.01)
        cash_advance_frequency = st.slider("Частота авансов наличными (Cash Advance Frequency)", 0.0, 1.0, 0.1, 0.01)
        
        st.subheader("Дополнительные параметры")
        cash_advance_trx = st.number_input("Количество операций с наличными (Cash Advance TRX)", min_value=0, value=0, step=1)
        purchases_trx = st.number_input("Количество покупок (Purchases TRX)", min_value=0, value=5, step=1)
        credit_limit = st.number_input("Кредитный лимит (Credit Limit)", min_value=0.0, value=5000.0, step=100.0)
        payments = st.number_input("Платежи (Payments)", min_value=0.0, value=300.0, step=50.0)
        minimum_payments = st.number_input("Минимальные платежи (Minimum Payments)", min_value=0.0, value=100.0, step=10.0)
        prc_full_payment = st.slider("Процент полных платежей (PRC Full Payment)", 0.0, 1.0, 0.3, 0.01)
        tenure = st.number_input("Срок обслуживания (Tenure)", min_value=0, value=12, step=1)
        
        submitted = st.form_submit_button("🎯 Классифицировать клиента")
    
    if submitted:
        st.header("Результаты ")
        
        input_data = pd.DataFrame({
            'BALANCE': [balance],
            'BALANCE_FREQUENCY': [balance_frequency],
            'PURCHASES': [purchases],
            'ONEOFF_PURCHASES': [oneoff_purchases],
            'INSTALLMENTS_PURCHASES': [installments_purchases],
            'CASH_ADVANCE': [cash_advance],
            'PURCHASES_FREQUENCY': [purchases_frequency],
            'ONEOFF_PURCHASES_FREQUENCY': [oneoff_purchases_frequency],
            'PURCHASES_INSTALLMENTS_FREQUENCY': [purchases_installments_frequency],
            'CASH_ADVANCE_FREQUENCY': [cash_advance_frequency],
            'CASH_ADVANCE_TRX': [cash_advance_trx],
            'PURCHASES_TRX': [purchases_trx],
            'CREDIT_LIMIT': [credit_limit],
            'PAYMENTS': [payments],
            'MINIMUM_PAYMENTS': [minimum_payments],
            'PRC_FULL_PAYMENT': [prc_full_payment],
            'TENURE': [tenure]
        })
        
        with st.expander("Просмотр введенных данных"):
            st.dataframe(input_data)
        
        try:
            scaled_data = scaler.transform(input_data)
            
            cluster = kmeans_model.predict(scaled_data)[0]
            
            cluster_info = get_cluster_description(cluster)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Результат")
                st.markdown(f"Сегмент: {cluster}")
                st.info(f"{cluster_info['name']}")
                
                st.subheader("Принадлежность к сегменту")
                segments = ["Консервативные", "Активные", "Сбалансированные"]
                for i, seg in enumerate(segments):
                    if i == cluster:
                        st.success(f"{seg} - ТЕКУЩИЙ СЕГМЕНТ")
                    else:
                        st.write(f"○ {seg}")
            
            with col2:
                st.subheader("📝 Описание сегмента")
                st.markdown(cluster_info['description'])
                
               
        
        except Exception as e:
            st.error(f"Ошибка при классификации: {str(e)}")
            
        
        except Exception as e:
            st.error(f"Ошибка при классификации: {str(e)}")
    
   
        
        
        
    
    

if __name__ == "__main__":
    main()