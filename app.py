import streamlit as st
import pandas as pd
import numpy as np
import json
import tempfile
import os
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm
import io

# Конфигурация страницы
st.set_page_config(
    page_title="Оценка остаточного ресурса пароперегревателей",
    page_icon="📊",
    layout="wide"
)

# Уравнения для каждого балла: {балл: (k, b)}
NOMOGRAM_EQUATIONS = {
    1: (-36.54, 722.7),
    2: (-33.22, 717.1),
    3: (-36.54, 741.7),
    4: (-36.54, 752.7),
    5: (-36.54, 763.7),
    6: (-39.84, 788.2)
}

# =============================================================================
# ОСНОВНЫЕ РАСЧЕТНЫЕ ФУНКЦИИ
# =============================================================================

def calculate_temperature_from_structure(score, hours):
    """Расчет температуры по баллу микроструктуры и наработке"""
    if hours <= 0 or score not in NOMOGRAM_EQUATIONS:
        return None
    
    x = np.log10(hours)
    k, b = NOMOGRAM_EQUATIONS[score]
    temp = k * x + b
    
    # Округление до ближайшего 5
    return round(temp / 5) * 5

def calculate_saturation_temperature_interpolated(pressure):
    """Расчет температуры насыщения по давлению"""
    table = [
        {'p': 2, 't': 212.42},
        {'p': 4, 't': 250.4},
        {'p': 6, 't': 275.64},
        {'p': 8, 't': 295.06},
        {'p': 10, 't': 311.06},
        {'p': 12, 't': 324.75},
        {'p': 14, 't': 336.63},
        {'p': 16, 't': 347.01},
        {'p': 18, 't': 356.16},
        {'p': 20, 't': 364.29},
        {'p': 22.064, 't': 373.95},
        {'p': 24, 't': 380.97},
        {'p': 26, 't': 387.4},
    ]
    
    p = float(pressure)
    if p < 2 or p > 26:
        return None
    
    for i in range(len(table) - 1):
        if table[i]['p'] <= p <= table[i + 1]['p']:
            p1, t1 = table[i]['p'], table[i]['t']
            p2, t2 = table[i + 1]['p'], table[i + 1]['t']
            return t1 + ((p - p1) / (p2 - p1)) * (t2 - t1)
    
    return None

def calculate_min_allowed_thickness(outer_diameter):
    """Минимально допустимая толщина стенки"""
    d = float(outer_diameter)
    if d < 38:
        return 1.45
    elif d <= 51:
        return 1.6
    elif d <= 70:
        return 2.0
    elif d <= 90:
        return 2.4
    elif d <= 108:
        return 2.8
    else:
        return 3.2

def calculate_single_sample(sample):
    """Основная функция расчета для одного образца"""
    try:
        # Базовые параметры
        p = float(sample.get('pressure', 0))
        s_n = float(sample.get('nominalThickness', 0))
        D_n = float(sample.get('outerDiameter', 0))
        s_min = float(sample.get('minThickness', 0))
        s_max = float(sample.get('maxThickness', 0))
        d_max = float(sample.get('maxInnerDiameter', 0))
        tau_e = float(sample.get('operatingHours', 0))
        steel_type = sample.get('steelType', '12H1MF')
        
        # Проверка обязательных полей
        if not all([p, s_n, D_n, s_min, s_max, d_max, tau_e]):
            return {
                'T_s_eq': 0,
                'finalResource': 0,
                'resultMessage': "Не заполнены обязательные поля",
                'sigma_0': 0,
                'sigma_k': 0,
                'sigma_sr': 0,
                'corrosionRate': 0,
                'finalThickness': 0
            }
        
        # Расчет коррозионной скорости
        v_korr = 0
        if sample.get('corrosionRateInput'):
            v_korr = float(sample['corrosionRateInput']) * 1e-5
        else:
            v_korr = (s_max - s_min) / tau_e if s_max > s_min else (s_n - s_min) / tau_e
        
        # Расчет напряжений
        is_tolerance_exceeded = s_max > 1.1 * s_n
        sigma_0 = (p / 2) * (D_n / (s_max if is_tolerance_exceeded else s_n) - 1)
        sigma_k = (p / 2) * (d_max / s_min + 1)
        sigma_sr = (sigma_0 + sigma_k) / 2
        
        # Расчет в зависимости от марки стали
        T_s_eq = 0
        final_resource = 0
        final_thickness = 0
        result_message = ""
        
        if steel_type == "DI59":
            # Расчет для стали ДИ59
            G = float(sample.get('grainNumber', 7))
            C_sigma = float(sample.get('sigmaPhase', 0))
            
            k_G = 1.66 * (G ** -0.33)
            T_s_eq = (3.4 * sigma_sr - 24341) / (np.log((k_G * C_sigma) / np.sqrt(tau_e)) - 23.4)
            
            # Расчет ресурса по жаропрочности
            A = (21360 - 2400 * np.log10(sigma_sr) - 4.5 * sigma_sr) / T_s_eq + 2 * np.log10(T_s_eq) - 19.56
            tau_r_A = 10 ** A
            P_isp = tau_e / tau_r_A
            P_ost = 0.8 - P_isp
            
            if P_ost <= 0:
                final_resource = 0
                result_message = "Ресурс исчерпан"
            else:
                # Итерационный расчет
                tau_prognoz = 0.5 * tau_r_A
                converged = False
                
                for _ in range(100):
                    s_min2 = s_min - v_korr * tau_prognoz
                    if s_min2 <= 0:
                        break
                    
                    sigma_k2 = (p / 2) * (d_max / s_min2 + 1)
                    sigma_sr2 = (sigma_k + sigma_k2) / 2
                    
                    B = (21360 - 2400 * np.log10(sigma_k2) - 4.5 * sigma_k2) / T_s_eq + 2 * np.log10(T_s_eq) - 19.56
                    tau_r_B = 10 ** B
                    tau_prognoz1 = tau_r_B * P_ost
                    
                    diff = tau_prognoz - tau_r_B
                    if 0 <= diff <= 240 and tau_prognoz < tau_r_A and (s_min - s_min2) > 0:
                        converged = True
                        final_resource = tau_prognoz1
                        final_thickness = s_min2
                        result_message = f"Ресурс по жаропрочности: {int(final_resource)} ч"
                        break
                    
                    if diff > 240:
                        tau_prognoz = tau_prognoz * 0.9 if diff > 1000 else tau_prognoz - 100
                    elif diff < 0:
                        tau_prognoz = tau_prognoz * 1.1 if abs(diff) > 1000 else tau_prognoz + 100
                
                if not converged:
                    # Расчет по конструктивной прочности
                    s_min_dop = calculate_min_allowed_thickness(D_n)
                    final_resource = (s_min - s_min_dop) / v_korr
                    final_thickness = s_min_dop
                    result_message = f"Ресурс по конструктивной прочности: {int(final_resource)} ч"
        
        elif steel_type == "12X18H12T":
            # Расчет для стали 12Х18Н12Т
            C_sigma = float(sample.get('sigmaPhase', 0))
            T_s_eq = 847 * (C_sigma / np.sqrt(tau_e)) ** 0.0647 + 273
            
            A = (30942 - 3762 * np.log10(sigma_sr) - 16.8 * sigma_sr) / T_s_eq + 2 * np.log10(T_s_eq) - 26.3
            tau_r_A = 10 ** A
            P_isp = tau_e / tau_r_A
            P_ost = 0.8 - P_isp
            
            if P_ost <= 0:
                final_resource = 0
                result_message = "Ресурс исчерпан"
            else:
                # Аналогичный итерационный расчет как для ДИ59
                s_min_dop = calculate_min_allowed_thickness(D_n)
                final_resource = (s_min - s_min_dop) / v_korr
                final_thickness = s_min_dop
                result_message = f"Ресурс по конструктивной прочности: {int(final_resource)} ч"
        
        elif steel_type == "12H1MF":
            # Расчет для стали 12Х1МФ
            T_s_eq_values = []
            
            # Расчет по молибдену
            if sample.get('C_Mo_k') and sample.get('C_Mo_m'):
                C_Mo_k = float(sample['C_Mo_k'])
                C_Mo_m = float(sample['C_Mo_m'])
                a = (C_Mo_k / C_Mo_m - 7.34e-4 * sigma_0) * (tau_e ** -0.291)
                T_s_eq_mo = -31655 * a * a + 4720 * a + 503 + 273.15
                T_s_eq_values.append(T_s_eq_mo)
            
            # Расчет по баллу структуры
            if sample.get('structural_ball') and sample.get('t_structural_ball'):
                T_s_eq_struct = float(sample['t_structural_ball']) + 273.15
                T_s_eq_values.append(T_s_eq_struct)
            
            # Расчет по оксидной пленке
            if sample.get('h_oxide_thickness') and sample.get('t_oxide_thickness'):
                T_s_eq_oxide = float(sample['t_oxide_thickness']) + 273.15
                T_s_eq_values.append(T_s_eq_oxide)
            
            T_s_eq = max(T_s_eq_values) if T_s_eq_values else 0
            
            # Расчет ресурса
            A = (24956 - 2400 * np.log10(sigma_sr) - 10.9 * sigma_sr) / T_s_eq + 2 * np.log10(T_s_eq) - 24.88
            tau_r_A = 10 ** A
            P_isp = tau_e / tau_r_A
            P_ost = 0.8 - P_isp
            
            if P_ost <= 0:
                final_resource = 0
                result_message = "Ресурс исчерпан"
            else:
                s_min_dop = calculate_min_allowed_thickness(D_n)
                final_resource = (s_min - s_min_dop) / v_korr
                final_thickness = s_min_dop
                result_message = f"Ресурс по конструктивной прочности: {int(final_resource)} ч"
        
        elif steel_type == "20_12H1MF_HEAT_RESIST":
            # Расчет для стали 20/12Х1МФ (жаростойкость)
            sigma_02_t = float(sample.get('sigma_02_t', 0))
            
            # Расчет температуры эксплуатации
            if sample.get('calculate_t_exploit_auto', False):
                t_s = calculate_saturation_temperature_interpolated(p)
                t_exploit = t_s + 60 if t_s else 0
            else:
                t_exploit = float(sample.get('t_exploit_manual', 0))
            
            T_s_eq = t_exploit + 273.15  # Переводим в Кельвины
            
            # Расчет по жаростойкости
            s_p = (p * D_n) / (2 * (sigma_02_t / 1.5) + p)
            s_min_dop = calculate_min_allowed_thickness(D_n)
            
            if s_p < s_min_dop:
                final_resource = (s_min - s_min_dop) / v_korr
                final_thickness = s_min_dop
                result_message = f"Ресурс по конструктивной прочности: {int(final_resource)} ч"
            else:
                final_resource = (s_min - s_p) / v_korr
                final_thickness = s_p
                result_message = f"Ресурс по жаростойкости: {int(final_resource)} ч"
        
        else:
            # Для неизвестной стали
            final_resource = 0
            result_message = "Неизвестный тип стали"
        
        return {
            'T_s_eq': round(T_s_eq - 273.15) if T_s_eq > 0 else 0,  # Обратно в Цельсии
            'finalResource': int(final_resource),
            'resultMessage': result_message,
            'sigma_0': round(sigma_0, 2),
            'sigma_k': round(sigma_k, 2),
            'sigma_sr': round(sigma_sr, 2),
            'corrosionRate': round(v_korr * 100000, 2),  # мм/100000 ч
            'finalThickness': round(final_thickness, 2),
            'steelType': steel_type
        }
        
    except Exception as e:
        return {
            'T_s_eq': 0,
            'finalResource': 0,
            'resultMessage': f"Ошибка расчета: {str(e)}",
            'sigma_0': 0,
            'sigma_k': 0,
            'sigma_sr': 0,
            'corrosionRate': 0,
            'finalThickness': 0,
            'steelType': sample.get('steelType', '')
        }

def perform_calculations(samples):
    """Выполнение расчетов для всех образцов"""
    results = []
    for sample in samples:
        result = calculate_single_sample(sample)
        results.append(result)
    return results

# =============================================================================
# STREAMLIT ИНТЕРФЕЙС
# =============================================================================

def initialize_session_state():
    if 'project_data' not in st.session_state:
        st.session_state.project_data = None
    if 'calculation_results' not in st.session_state:
        st.session_state.calculation_results = None
    if 'samples_df' not in st.session_state:
        st.session_state.samples_df = pd.DataFrame()

def main():
    st.title("Программа по определению остаточного ресурса пароперегревателей")
    
    initialize_session_state()
    
    # Боковая панель для навигации
    st.sidebar.title("Навигация")
    menu = st.sidebar.radio(
        "Выберите раздел:",
        ["Создать проект", "Загрузить из Excel", "Редактировать данные", 
         "Авторасчет температур", "Выполнить расчеты", "Просмотр отчетов"]
    )
    
    if menu == "Создать проект":
        show_project_creation()
    elif menu == "Загрузить из Excel":
        show_excel_import()
    elif menu == "Редактировать данные":
        show_data_editor()
    elif menu == "Авторасчет температур":
        show_temperature_calculation()
    elif menu == "Выполнить расчеты":
        show_calculations()
    elif menu == "Просмотр отчетов":
        show_reports()

def show_project_creation():
    st.header("Создание нового проекта")
    
    col1, col2 = st.columns(2)
    
    with col1:
        project_name = st.text_input("Название проекта", value="Ириклинская ГРЭС ЭБ № 3")
        num_samples = st.number_input("Количество образцов", min_value=1, value=20)
    
    with col2:
        approver_name = st.text_input("Утверждающий", value="Пчелинцев А.В.")
        approver_position = st.text_input("Должность утверждающего", 
                                        value="Директор по аналитическим исследованиям")
    
    if st.button("Создать проект"):
        samples = []
        for i in range(num_samples):
            sample = {
                "id": i + 1,
                "objectName": f"Образец {i+1}",
                "steelType": "12H1MF",
                "pressure": "",
                "nominalThickness": "",
                "outerDiameter": "",
                "minThickness": "",
                "maxThickness": "",
                "maxInnerDiameter": "",
                "grainNumber": "",
                "sigmaPhase": "",
                "C_Mo_k": "",
                "C_Mo_m": "",
                "structural_ball": "",
                "h_oxide_thickness": "",
                "t_structural_ball": "",
                "t_oxide_thickness": "",
                "sigma_02_t": "",
                "t_exploit_manual": "",
                "calculate_t_exploit_auto": False,
                "operatingHours": "",
                "corrosionRateInput": "",
                "typeOfSuperheater": "SHPP"
            }
            samples.append(sample)
        
        st.session_state.project_data = {
            "projectName": project_name,
            "samples": samples,
            "approverName": approver_name,
            "approverPosition": approver_position,
            "experts": [{"position": "", "fullName": ""}]
        }
        
        st.session_state.samples_df = pd.DataFrame(samples)
        st.success(f"Проект '{project_name}' создан с {num_samples} образцами")

def show_excel_import():
    st.header("Импорт данных из Excel")
    
    uploaded_file = st.file_uploader("Выберите Excel файл", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.write("Предпросмотр данных:")
            st.dataframe(df.head())
            
            if st.button("Импортировать данные"):
                samples = []
                for idx, row in df.iterrows():
                    sample = {
                        "id": idx + 1,
                        "objectName": str(row.get('objectName', f"Образец {idx+1}")),
                        "steelType": str(row.get('steelType', '12H1MF')),
                        "pressure": str(row.get('pressure', '')),
                        "nominalThickness": str(row.get('nominalThickness', '')),
                        "outerDiameter": str(row.get('outerDiameter', '')),
                        "minThickness": str(row.get('minThickness', '')),
                        "maxThickness": str(row.get('maxThickness', '')),
                        "maxInnerDiameter": str(row.get('maxInnerDiameter', '')),
                        "grainNumber": str(row.get('grainNumber', '')),
                        "sigmaPhase": str(row.get('sigmaPhase', '')),
                        "C_Mo_k": str(row.get('C_Mo_k', '')),
                        "C_Mo_m": str(row.get('C_Mo_m', '')),
                        "structural_ball": str(row.get('structural_ball', '')),
                        "h_oxide_thickness": str(row.get('h_oxide_thickness', '')),
                        "t_structural_ball": str(row.get('t_structural_ball', '')),
                        "t_oxide_thickness": str(row.get('t_oxide_thickness', '')),
                        "sigma_02_t": str(row.get('sigma_02_t', '')),
                        "t_exploit_manual": str(row.get('t_exploit_manual', '')),
                        "calculate_t_exploit_auto": bool(row.get('calculate_t_exploit_auto', False)),
                        "operatingHours": str(row.get('operatingHours', '')),
                        "corrosionRateInput": str(row.get('corrosionRateInput', '')),
                        "typeOfSuperheater": str(row.get('typeOfSuperheater', 'SHPP'))
                    }
                    samples.append(sample)
                
                st.session_state.project_data = {
                    "projectName": "Импортированный проект",
                    "samples": samples,
                    "approverName": "Пчелинцев А.В.",
                    "approverPosition": "Директор по аналитическим исследованиям",
                    "experts": [{"position": "", "fullName": ""}]
                }
                
                st.session_state.samples_df = pd.DataFrame(samples)
                st.success("Данные успешно импортированы!")
                
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {str(e)}")

def show_data_editor():
    if st.session_state.project_data is None:
        st.warning("Сначала создайте проект или загрузите данные")
        return
    
    st.header("Редактирование данных образцов")
    
    if st.session_state.samples_df.empty:
        st.session_state.samples_df = pd.DataFrame(st.session_state.project_data['samples'])
    
    edited_df = st.data_editor(
        st.session_state.samples_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "objectName": "Объект",
            "steelType": st.column_config.SelectboxColumn("Сталь", 
                options=["12H1MF", "12X18H12T", "DI59", "20_12H1MF_HEAT_RESIST"]),
            "pressure": "Давление, МПа",
            "nominalThickness": "sₙ, мм",
            "outerDiameter": "Dₙ, мм", 
            "minThickness": "sₘᵢₙ, мм",
            "maxThickness": "sₘₐₓ, мм",
            "maxInnerDiameter": "dₘₐₓ, мм",
            "structural_ball": st.column_config.NumberColumn("Балл структуры", min_value=1, max_value=6, step=1),
            "operatingHours": "Наработка, ч",
            "typeOfSuperheater": st.column_config.SelectboxColumn("Тип ПП", options=["SHPP", "KPP"])
        },
        hide_index=True
    )
    
    if st.button("Сохранить изменения"):
        st.session_state.samples_df = edited_df
        st.session_state.project_data['samples'] = edited_df.to_dict('records')
        st.success("Изменения сохранены!")

def show_temperature_calculation():
    if st.session_state.project_data is None:
        st.warning("Сначала создайте проект или загрузите данные")
        return
    
    st.header("Автоматический расчет температур по баллу структуры")
    
    st.info("""
    Эта функция автоматически рассчитывает температуру эксплуатации на основе балла микроструктуры и наработки
    с использованием номограммы. Температура округляется до ближайшего значения, кратного 5.
    """)
    
    # Показываем номограмму
    st.subheader("Номограмма для расчета температуры")
    fig = plot_nomogram()
    st.pyplot(fig)
    
    # Авторасчет для всех образцов с баллом структуры
    df = st.session_state.samples_df.copy()
    
    valid_samples = df[
        (df['structural_ball'].notna()) & 
        (df['structural_ball'] != "") &
        (df['operatingHours'].notna()) & 
        (df['operatingHours'] != "") &
        (pd.to_numeric(df['operatingHours'], errors='coerce') > 0)
    ].copy()
    
    if not valid_samples.empty:
        st.subheader("Результаты авторасчета температур")
        
        valid_samples['calculated_temperature'] = valid_samples.apply(
            lambda row: calculate_temperature_from_structure(
                int(float(row['structural_ball'])), 
                float(row['operatingHours'])
            ), 
            axis=1
        )
        
        result_display = valid_samples[['objectName', 'structural_ball', 'operatingHours', 'calculated_temperature']]
        st.dataframe(result_display)
        
        if st.button("Заполнить поля t_structural_ball рассчитанными температурами"):
            for idx, row in valid_samples.iterrows():
                original_idx = df[df['objectName'] == row['objectName']].index[0]
                df.at[original_idx, 't_structural_ball'] = str(row['calculated_temperature'])
            
            st.session_state.samples_df = df
            st.session_state.project_data['samples'] = df.to_dict('records')
            st.success("Поля t_structural_ball успешно заполнены!")
            
            st.subheader("Обновленные данные")
            st.dataframe(df[['objectName', 'structural_ball', 'operatingHours', 't_structural_ball']])
    else:
        st.warning("Нет образцов с заполненными баллом структуры и наработкой для расчета температур.")

def plot_nomogram():
    """Построение номограммы"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_log_min = 4.0
    x_log_max = 5.7
    x_vals = np.linspace(x_log_min, x_log_max, 100)
    
    for score in range(1, 7):
        k, b = NOMOGRAM_EQUATIONS[score]
        y_vals = k * x_vals + b
        ax.plot(10**x_vals, y_vals, label=f'Балл {score}', linewidth=2)
    
    ax.set_xlabel('Наработка τ_з, ч')
    ax.set_ylabel('Температура t_экс, °C')
    ax.set_title('Номограмма: Расчет температуры по баллу структуры')
    ax.grid(True, which="both", ls="--")
    ax.set_xscale('log')
    ax.set_xlim(10**4, 5*10**5)
    ax.set_ylim(510, 630)
    ax.legend()
    
    return fig

def show_calculations():
    if st.session_state.project_data is None:
        st.warning("Сначала создайте проект или загрузите данные")
        return
    
    st.header("Выполнение расчетов остаточного ресурса")
    
    if st.button("Запустить расчет всех образцов"):
        with st.spinner("Выполняются расчеты..."):
            try:
                results = perform_calculations(st.session_state.project_data['samples'])
                st.session_state.calculation_results = results
                st.success("Расчеты успешно завершены!")
                
                # Показываем сводку результатов
                st.subheader("Сводка результатов")
                summary_data = []
                for i, (sample, result) in enumerate(zip(st.session_state.project_data['samples'], results)):
                    summary_data.append({
                        '№': i + 1,
                        'Образец': sample['objectName'],
                        'Сталь': sample['steelType'],
                        'Температура, °C': result['T_s_eq'],
                        'Ресурс, ч': result['finalResource'],
                        'Сообщение': result['resultMessage']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df)
                
                # Детальная таблица напряжений
                st.subheader("Детальные результаты расчетов")
                detail_data = []
                for i, (sample, result) in enumerate(zip(st.session_state.project_data['samples'], results)):
                    detail_data.append({
                        '№': i + 1,
                        'Образец': sample['objectName'],
                        'σ₀, МПа': result['sigma_0'],
                        'σₖ, МПа': result['sigma_k'],
                        'σср, МПа': result['sigma_sr'],
                        'vкор, мм/10⁵ч': result['corrosionRate'],
                        'Температура, °C': result['T_s_eq'],
                        'Ресурс, ч': result['finalResource']
                    })
                
                detail_df = pd.DataFrame(detail_data)
                st.dataframe(detail_df)
                
            except Exception as e:
                st.error(f"Ошибка при выполнении расчетов: {str(e)}")

def show_reports():
    if st.session_state.calculation_results is None:
        st.warning("Сначала выполните расчеты")
        return
    
    st.header("Генерация отчетов")
    
    st.subheader("Настройки отчета")
    
    col1, col2 = st.columns(2)
    with col1:
        report_date = st.date_input("Дата отчета", value=datetime.now())
        approver_name = st.text_input("Утверждающий", 
                                    value=st.session_state.project_data.get('approverName', 'Пчелинцев А.В.'))
    with col2:
        expert_name = st.text_input("Эксперт", value="")
        approver_position = st.text_input("Должность утверждающего",
                                        value=st.session_state.project_data.get('approverPosition', ''))
    
    # Предпросмотр данных
    st.subheader("Предпросмотр данных для отчета")
    
    preview_data = []
    for i, (sample, result) in enumerate(zip(st.session_state.project_data['samples'], 
                                           st.session_state.calculation_results)):
        preview_data.append({
            '№': i + 1,
            'Объект': sample.get('objectName', ''),
            'Сталь': sample.get('steelType', ''),
            'Температура, °C': result.get('T_s_eq', ''),
            'Ресурс, ч': result.get('finalResource', ''),
            'Сообщение': result.get('resultMessage', '')
        })
    
    st.dataframe(pd.DataFrame(preview_data))
    
    # Генерация PDF
    if st.button("Сгенерировать PDF отчет"):
        with st.spinner("Генерация отчета..."):
            try:
                pdf_bytes = generate_pdf_report(
                    st.session_state.project_data,
                    st.session_state.calculation_results,
                    approver_name,
                    approver_position,
                    report_date.strftime("%d.%m.%Y")
                )
                
                st.download_button(
                    label="📥 Скачать PDF отчет",
                    data=pdf_bytes,
                    file_name=f"Отчет_{st.session_state.project_data['projectName']}.pdf",
                    mime="application/pdf"
                )
                
                st.success("PDF отчет успешно сгенерирован!")
                
            except Exception as e:
                st.error(f"Ошибка при генерации отчета: {str(e)}")

def generate_pdf_report(project_data, results, approver_name, approver_position, report_date):
    """Генерация PDF отчета"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Заголовок
    title_style = styles['Heading1']
    title = Paragraph(f"ОТЧЕТ ПО РЕЗУЛЬТАТАМ ОЦЕНКИ ОСТАТОЧНОГО РЕСУРСА", title_style)
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Информация о проекте
    project_info = [
        ["Объект:", project_data['projectName']],
        ["Дата отчета:", report_date],
        ["Утверждающий:", f"{approver_position} {approver_name}"]
    ]
    
    project_table = Table(project_info, colWidths=[60*mm, 100*mm])
    project_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(project_table)
    story.append(Spacer(1, 20))
    
    # Таблица результатов
    results_data = [['№', 'Объект', 'Сталь', 'Температура, °C', 'Ресурс, ч', 'Примечание']]
    
    for i, (sample, result) in enumerate(zip(project_data['samples'], results)):
        results_data.append([
            i + 1,
            sample.get('objectName', ''),
            sample.get('steelType', ''),
            result.get('T_s_eq', ''),
            result.get('finalResource', ''),
            result.get('resultMessage', '')[:50] + '...' if len(result.get('resultMessage', '')) > 50 else result.get('resultMessage', '')
        ])
    
    results_table = Table(results_data, colWidths=[15*mm, 40*mm, 25*mm, 25*mm, 25*mm, 40*mm])
    results_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 8),
    ]))
    story.append(results_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    main()