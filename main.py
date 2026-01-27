# 导入必要的库
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import io
from datetime import datetime
import math

# 设置页面标题和布局
st.set_page_config(
    page_title="CMM-SCREEN: Community Batch Risk Assessment System",
    layout="wide",
    page_icon="🏥"
)

# 添加CSS样式
st.markdown("""
<style>
    /* 主卡片样式 */
    .main-card {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid #3498db;
    }
    
    .instruction-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        border: 2px dashed #dee2e6;
    }
    
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-top: 5px solid #28a745;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #ffc107;
    }
    
    .variable-table {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 15px 0;
        font-size: 14px;
    }
    
    .section-header {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 10px 10px 0 0;
        margin-top: 30px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# 模型文件路径
MODEL_PATH = "LR_model.sav"


# === 连续变量归一化参数 ===
feature_mins = np.array([
    65.0, 35.0, 36.0, 0.1, 0.1, 36.0, 50.0, 41.0, 24.0, 1.0,
    1.0, 0.33, 0.1326, 78.0, 50.0, -0.285714285714278,
    7.22076951556516, 120.347713086439, 1.0
])

feature_maxs = np.array([
    112.0, 39.0, 160.0, 2.0, 2.0, 126.0, 234.0, 906.0, 567.0, 48.8,
    11.89, 7.45, 5.0, 208.0, 133.0, 51.5384615384615,
    11.3181415603046, 595.622915922451, 3.0
])

# 基础特征名称（用户需要提供的）
base_feature_names = [
    'age', 'gender', 'temperature', 'pulse', 'systolic', 'diastolic',
    'left_naked_eye', 'right_naked_eye', 'heart_rate', 'hear', 'exercise', 'heart_rhythm',
    'smoke', 'HGB', 'PLT', 'Creatinine', 'Urea', 'TC', 'LDL_C', 'HDL_C',
    'hypertension_final', 'diabetes_final', 'dyslipidemia_final',
    'height_cm', 'waist_cm', 'weight_kg', 'glucose_mmolL', 'triglycerides_mmolL'
]

# 衍生特征名称（系统自动计算）
derived_feature_names = ['RFM', 'TyG', 'TyG_BMI', 'count']

# 所有模型需要的特征（基础+衍生）
all_feature_names_for_model = [
    'age', 'temperature', 'pulse', 'left_naked_eye', 'right_naked_eye', 'heart_rate',
    'HGB', 'PLT', 'Creatinine', 'Urea', 'TC', 'LDL_C', 'HDL_C', 'systolic', 'diastolic',
    'RFM', 'TyG', 'TyG_BMI', 'count',
    'gender', 'smoke', 'heart_rhythm', 'hear', 'exercise'
]

# 特征英文描述（供用户参考）
feature_descriptions = {
    # 基础特征
    'age': 'Age (years)',
    'gender': 'Gender (0=Male, 1=Female)',
    'temperature': 'Body Temperature (°C)',
    'pulse': 'Pulse rate (beats/min)',
    'systolic': 'Systolic Blood Pressure (mmHg)',
    'diastolic': 'Diastolic Blood Pressure (mmHg)',
    'left_naked_eye': 'Left eye visual acuity (decimal notation: 0.1-2.0)',
    'right_naked_eye': 'Right eye visual acuity (decimal notation: 0.1-2.0)',
    'heart_rate': 'Heart Rate (beats/min)',
    'hear': 'Hearing ability (0=Normal, 1=Abnormal)',
    'exercise': 'Motor function (0=Normal, 1=Abnormal)',
    'heart_rhythm': 'Heart rhythm (0=Regular, 1=Irregular)',
    'smoke': 'Smoking status (0=Non-smoker, 1=Smoker)',
    'HGB': 'Hemoglobin (g/L)',
    'PLT': 'Platelet count (×10⁹/L)',
    'Creatinine': 'Creatinine (μmol/L)',
    'Urea': 'Urea (mmol/L)',
    'TC': 'Total Cholesterol (mmol/L)',
    'LDL_C': 'Low-density lipoprotein (mmol/L)',
    'HDL_C': 'High-density lipoprotein (mmol/L)',
    'hypertension_final': 'Hypertension (0=No, 1=Yes)',
    'diabetes_final': 'Diabetes (0=No, 1=Yes)',
    'dyslipidemia_final': 'Dyslipidemia (0=No, 1=Yes)',
    'height_cm': 'Height (cm)',
    'waist_cm': 'Waist circumference (cm)',
    'weight_kg': 'Weight (kg)',
    'glucose_mmolL': 'Glucose (mmol/L)',
    'triglycerides_mmolL': 'Triglycerides (mmol/L)',
    
    # 衍生特征（系统自动计算）
    'count': 'Number of existing diseases (auto-calculated)',
    'RFM': 'Relative Fat Mass index (auto-calculated)',
    'TyG': 'Triglyceride-glucose index (auto-calculated)',
    'TyG_BMI': 'TyG-BMI index (auto-calculated)'
}

def calculate_derived_features(df):
    """
    计算衍生特征
    """
    # 计算疾病数量
    df['count'] = df['hypertension_final'] + df['diabetes_final'] + df['dyslipidemia_final']
    
    # 计算RFM（根据性别：0=男，1=女）
    # RFM = 64 - 20 * (身高/腰围) + 12 * 性别（男性=0，女性=1）
    df['RFM'] = 64 - 20 * (df['height_cm'] / df['waist_cm']) + 12 * df['gender']
    
    # 计算TyG指数：ln[TG(mg/dL) × GLU(mg/dL) / 2]
    # 根据你的R代码，转换系数为：
    # TG: 1 mmol/L = 88.6 mg/dL
    # GLU: 1 mmol/L = 18.0 mg/dL
    TG_mg_dL = df['triglycerides_mmolL'] * 88.6  # 修正：使用88.6
    GLU_mg_dL = df['glucose_mmolL'] * 18.0       # 修正：使用18.0
    df['TyG'] = np.log(TG_mg_dL * GLU_mg_dL / 2)
    
    # 计算BMI
    df['BMI'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    
    # 计算TyG-BMI
    df['TyG_BMI'] = df['TyG'] * df['BMI']
    
    return df

@st.cache_resource
def load_model():
    """
    加载LR模型
    """
    try:
        # 方法1：优先使用joblib加载（因为你是用joblib.dump保存的）
        model = joblib.load(MODEL_PATH)
        st.success("✅ 模型加载成功 (使用joblib)")
        
        # 创建标准化的返回格式
        model_content = {
            'model': model,
            'best_threshold': 0.435
        }
        return model, model_content
            
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

def main():
    """
    主函数
    """
    # 主标题
    st.markdown("""
    <div class="main-card">
        <div style="text-align: center;">
            <h1 style="color: white; margin-bottom: 15px; font-size: 2.8em;">🏥 CMM-SCREEN</h1>
            <h3 style="color: white; opacity: 0.95; margin-bottom: 10px;">Community Batch Cardiometabolic Multimorbidity Risk Assessment System</h3>
            <p style="color: white; opacity: 0.85; font-size: 1.1em;">Batch risk assessment tool designed for public health workers</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 创建标签页
    tab1, tab2 = st.tabs(["📊 Batch Risk Assessment", "📋 Data Requirements"])
    
    with tab1:
        show_batch_prediction()
    
    with tab2:
        show_data_requirements()

def show_batch_prediction():
    """
    显示批量预测界面
    """
    st.markdown('<div class="section-header"><h3 style="margin: 0;">🚀 Quick Start: Batch Risk Assessment</h3></div>', unsafe_allow_html=True)
    
    # 使用说明
    st.markdown("""
    <div class="instruction-card">
        <h4>📝 How to use:</h4>
        <ol>
            <li><strong>Download template</strong>: Get the CSV template file with all required basic variables</li>
            <li><strong>Prepare data</strong>: Fill in community health data according to the template format</li>
            <li><strong>Upload data</strong>: Upload the prepared data file</li>
            <li><strong>Run assessment</strong>: System automatically calculates derived indices and performs batch risk assessment</li>
            <li><strong>Download report</strong>: Get detailed report with risk assessment results and all calculated indices</li>
        </ol>
        <p style="color: #666; margin-top: 10px; font-size: 0.9em;">
            <strong>Note:</strong> This system is only for cardiometabolic multimorbidity risk assessment in people aged 65 and above.
            <strong>System will automatically calculate RFM, TyG, TyG-BMI, and disease count.</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card"><h4>📥 Step 1: Get Data Template</h4>', unsafe_allow_html=True)
        
        # 创建示例数据 - 包含所有基础变量，加上ID列
        all_columns = ['ID'] + base_feature_names  # 添加ID列
        
        sample_data = pd.DataFrame(columns=all_columns)
        
        # 添加一行示例数据
        example_row = {'ID': 'EXAMPLE001'}  # 添加示例ID
        
        for feature in base_feature_names:
            if feature == 'age':
                example_row[feature] = 70.0
            elif feature in ['temperature', 'pulse', 'heart_rate']:
                example_row[feature] = 70.0 if feature == 'age' else 36.5
            elif feature in ['left_naked_eye', 'right_naked_eye']:
                example_row[feature] = 1.0
            elif feature in ['height_cm']:
                example_row[feature] = 165.0
            elif feature in ['waist_cm']:
                example_row[feature] = 85.0
            elif feature in ['weight_kg']:
                example_row[feature] = 65.0
            elif feature in ['glucose_mmolL', 'triglycerides_mmolL']:
                example_row[feature] = 5.5 if feature == 'glucose_mmolL' else 1.5
            elif feature in ['gender', 'smoke', 'heart_rhythm', 'hear', 'exercise',
                           'hypertension_final', 'diabetes_final', 'dyslipidemia_final']:
                example_row[feature] = 0
            elif feature in ['systolic', 'diastolic']:
                example_row[feature] = 120.0 if feature == 'systolic' else 80.0
            elif feature in ['HGB', 'PLT']:
                example_row[feature] = 135.0 if feature == 'HGB' else 250.0
            elif feature in ['Creatinine', 'Urea']:
                example_row[feature] = 70.0 if feature == 'Creatinine' else 5.0
            elif feature in ['TC', 'LDL_C', 'HDL_C']:
                example_row[feature] = 4.5 if feature == 'TC' else (2.5 if feature == 'LDL_C' else 1.2)
            else:
                example_row[feature] = 0.0
        
        sample_data = pd.DataFrame([example_row])
        
        # 转换为CSV
        csv_data = sample_data.to_csv(index=False, encoding='utf-8-sig')
        
        st.download_button(
            label="📋 Download CSV Template",
            data=csv_data,
            file_name="CMM_Risk_Assessment_Template.csv",
            mime="text/csv",
            key="template_csv",
            use_container_width=True
        )
        
        st.markdown("""
        <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        <p style="font-size: 0.9em; margin: 0;">
        <strong>Template includes:</strong>
        <ul style="margin: 5px 0 0 0;">
        <li>ID column for record identification (not used in prediction)</li>
        <li>27 basic variables you need to provide</li>
        <li>One example row with typical values</li>
        <li>System will automatically calculate derived indices</li>
        </ul>
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card"><h4>📁 Step 2: Upload Data File</h4>', unsafe_allow_html=True)
        
        # 文件上传器
        uploaded_file = st.file_uploader(
            "Choose data file",
            type=['csv'],
            help="Supports CSV format only. Please ensure data columns match the template"
        )
        
        if uploaded_file is not None:
            try:
                # 读取CSV文件
                data = pd.read_csv(uploaded_file)
                
                # 检查是否包含ID列
                has_id_column = 'ID' in data.columns
                
                # 显示数据预览
                st.success(f"✅ Successfully loaded file: {uploaded_file.name}")
                
                if has_id_column:
                    st.info(f"✓ ID column found: {len(data['ID'].unique())} unique IDs")
                    # 保存ID列，然后从数据中移除（不用于预测）
                    id_data = data['ID'].copy() if 'ID' in data.columns else None
                    data = data.drop(columns=['ID']) if 'ID' in data.columns else data
                else:
                    st.warning("⚠️ No ID column found. Consider adding ID for record management.")
                
                st.info(f"Data dimensions: {data.shape[0]} rows × {data.shape[1]} columns")
                
                with st.expander("👁️ Preview first 5 rows", expanded=True):
                    display_cols = ['ID'] + list(data.columns[:5]) if has_id_column else data.columns[:6]
                    st.dataframe(data.head(), use_container_width=True)
                
                # 检查按钮
                if st.button("🚀 Run Batch Risk Assessment", type="primary", use_container_width=True):
                    perform_batch_prediction(data, id_data if has_id_column else None)
                    
            except Exception as e:
                st.error(f"❌ File reading failed: {str(e)}")
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.write("**Possible issues:**")
                st.write("1. Incorrect file format")
                st.write("2. File encoding issues (recommend UTF-8 encoding)")
                st.write("3. Missing required columns")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Please upload a data file first")
        
        st.markdown('</div>', unsafe_allow_html=True)

def perform_batch_prediction(data, id_data=None):
    """
    执行批量预测
    """
    # ===== 添加这部分代码 =====
    # 在函数内部加载模型
    model_result = load_model()
    if model_result[0] is None:
        st.error("❌ Model not loaded correctly, cannot perform prediction")
        return
    
    model, model_content = model_result
    custom_threshold = model_content.get('best_threshold', 0.3)
    # ===== 添加结束 =====
    
    with st.spinner('🔍 Checking data integrity...'):
        # 检查必需的列
        missing_columns = [col for col in base_feature_names if col not in data.columns]
        
        if missing_columns:
            st.error(f"❌ Data missing required columns: {', '.join(missing_columns)}")
            
            # 显示缺失列的详细信息
            missing_info = pd.DataFrame({
                'Missing Variable': missing_columns,
                'Description': [feature_descriptions.get(col, 'Unknown') for col in missing_columns]
            })
            st.dataframe(missing_info, use_container_width=True)
            return
        
        # 检查数据类型
        try:
            # 转换数据类型
            for col in base_feature_names:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 检查缺失值
            missing_count = data[base_feature_names].isnull().sum().sum()
            if missing_count > 0:
                st.warning(f"⚠️ Found {missing_count} missing values, will fill with column means")
                data[base_feature_names] = data[base_feature_names].fillna(data[base_feature_names].mean())
        
        except Exception as e:
            st.error(f"❌ Data processing failed: {str(e)}")
            return
    
    # 计算衍生特征
    with st.spinner('🧮 Calculating derived indices...'):
        try:
            data_with_derived = calculate_derived_features(data.copy())
            
            # 显示计算出的衍生特征
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📈 Automatically Calculated Indices")
            
            # 显示前5行的衍生特征
            derived_cols = derived_feature_names + ['BMI']  # 也显示中间计算的BMI
            derived_preview = data_with_derived[derived_cols].head()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 5 rows of calculated indices:**")
                st.dataframe(derived_preview, use_container_width=True)
            
            with col2:
                # 显示统计摘要
                st.write("**Summary statistics of calculated indices:**")
                summary_stats = data_with_derived[derived_feature_names].describe().round(3)
                st.dataframe(summary_stats, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Failed to calculate derived features: {str(e)}")
            return
    
    # 执行预测
    with st.spinner('🤖 Performing risk assessment...'):
        try:
            # 准备模型输入数据
            # 确保我们有所有需要的特征
            for feature in all_feature_names_for_model:
                if feature not in data_with_derived.columns:
                    st.error(f"❌ Missing calculated feature: {feature}")
                    return
            
            # 提取模型需要的特征
            model_input_data = data_with_derived[all_feature_names_for_model].copy()
            
            # 分离连续变量和分类变量
            continuous_features_model = [
                'age', 'temperature', 'pulse', 'left_naked_eye', 'right_naked_eye', 'heart_rate',
                'HGB', 'PLT', 'Creatinine', 'Urea', 'TC', 'LDL_C', 'HDL_C', 'systolic', 'diastolic',
                'RFM', 'TyG', 'TyG_BMI', 'count'
            ]
            
            categorical_features_model = [
                'gender', 'smoke', 'heart_rhythm', 'hear', 'exercise'
            ]
            
            batch_continuous = model_input_data[continuous_features_model].values
            batch_categorical = model_input_data[categorical_features_model].values
            
            # 归一化连续变量
            batch_normalized = (batch_continuous - feature_mins) / (feature_maxs - feature_mins)
            batch_normalized = np.clip(batch_normalized, 0, 1)
            
            # 合并特征
            batch_features = np.concatenate([batch_normalized, batch_categorical], axis=1)
            batch_features = batch_features.astype(np.float64)
            
            # 使用模型预测
            if hasattr(model, 'predict_proba'):
                batch_probabilities = model.predict_proba(batch_features)
                batch_high_risk_probs = batch_probabilities[:, 1]
                batch_predictions = (batch_high_risk_probs >= custom_threshold).astype(int)
            else:
                batch_predictions = model.predict(batch_features)
                batch_high_risk_probs = np.zeros_like(batch_predictions, dtype=float)
            
            # 创建完整结果数据
            result_data = data_with_derived.copy()
            
            # 如果有ID数据，添加到结果中
            if id_data is not None:
                result_data.insert(0, 'ID', id_data)
            
            result_data['Risk_Assessment_Level'] = batch_predictions
            result_data['Risk_Assessment_Level'] = result_data['Risk_Assessment_Level'].map({0: 'Low Risk', 1: 'High Risk'})
            
            if hasattr(model, 'predict_proba'):
                result_data['Low_Risk_Probability'] = batch_probabilities[:, 0].round(4)
                result_data['High_Risk_Probability'] = batch_probabilities[:, 1].round(4)
            
            # 添加评估日期
            result_data['Assessment_Date'] = datetime.now().strftime('%Y-%m-%d')
            
            # 显示结果
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            # 结果概览
            col1, col2, col3 = st.columns(3)
            
            total_count = len(result_data)
            high_risk_count = (batch_predictions == 1).sum()
            high_risk_percent = (high_risk_count / total_count * 100) if total_count > 0 else 0
            
            with col1:
                st.metric("📊 Total Assessed", f"{total_count:,}")
            with col2:
                st.metric("⚠️ High Risk Count", f"{high_risk_count:,}")
            with col3:
                st.metric("📈 High Risk Percentage", f"{high_risk_percent:.1f}%")
            
            # 风险分布可视化
            st.subheader("📊 Risk Distribution")
            risk_counts = result_data['Risk_Assessment_Level'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # 使用streamlit原生图表
                chart_data = pd.DataFrame({
                    'Risk Level': risk_counts.index,
                    'Count': risk_counts.values
                })
                st.bar_chart(chart_data.set_index('Risk Level'))
            
            with col2:
                st.dataframe(risk_counts, use_container_width=True)
            
            # 详细结果表格
            st.subheader("📋 Detailed Assessment Results")
            
            # 选择要显示的列（优先显示ID）
            display_columns = []
            if 'ID' in result_data.columns:
                display_columns.append('ID')
            
            display_columns.extend(['age', 'gender', 'Risk_Assessment_Level'])
            
            if hasattr(model, 'predict_proba'):
                display_columns.extend(['Low_Risk_Probability', 'High_Risk_Probability'])
            
            # 添加一些关键衍生特征
            display_columns.extend(['RFM', 'TyG', 'count'])
            
            with st.expander("View assessment results", expanded=True):
                display_df = result_data[display_columns].copy()
                # 重命名列以便更好显示
                column_rename = {
                    'ID': 'ID',
                    'age': 'Age',
                    'gender': 'Gender',
                    'Risk_Assessment_Level': 'Risk Level',
                    'Low_Risk_Probability': 'Low Risk Prob',
                    'High_Risk_Probability': 'High Risk Prob',
                    'RFM': 'RFM Index',
                    'TyG': 'TyG Index',
                    'count': 'Disease Count'
                }
                display_df = display_df.rename(columns=column_rename)
                st.dataframe(display_df, use_container_width=True)
            
            # 下载按钮
            st.subheader("💾 Download Assessment Report")
            
            # CSV下载 - 包含所有数据
            csv_data = result_data.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 Download Full CSV Report",
                data=csv_data,
                file_name=f"CMM_Risk_Assessment_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 高风险人群列表
            if high_risk_count > 0:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.subheader("⚠️ High Risk Population (Recommend Priority Intervention)")
                
                high_risk_data = result_data[result_data['Risk_Assessment_Level'] == 'High Risk']
                
                # 显示关键信息
                priority_columns = []
                if 'ID' in high_risk_data.columns:
                    priority_columns.append('ID')
                
                priority_columns.extend(['age', 'gender', 'High_Risk_Probability', 'count', 
                                       'hypertension_final', 'diabetes_final', 'dyslipidemia_final'])
                
                priority_df = high_risk_data[priority_columns].copy()
                priority_df = priority_df.rename(columns={
                    'ID': 'ID',
                    'age': 'Age',
                    'gender': 'Gender',
                    'High_Risk_Probability': 'Risk Probability',
                    'count': 'Disease Count',
                    'hypertension_final': 'Hypertension',
                    'diabetes_final': 'Diabetes',
                    'dyslipidemia_final': 'Dyslipidemia'
                })
                
                # 映射0/1为Yes/No
                for col in ['Hypertension', 'Diabetes', 'Dyslipidemia']:
                    priority_df[col] = priority_df[col].map({0: 'No', 1: 'Yes'})
                
                st.dataframe(priority_df, use_container_width=True)
                
                st.info(f"**Total high-risk individuals identified: {high_risk_count}**")
                st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Prediction process error: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")

def show_data_requirements():
    """
    显示数据要求说明
    """
    st.markdown('<div class="section-header"><h3 style="margin: 0;">📋 Data Requirements and Automatic Calculations</h3></div>', unsafe_allow_html=True)
    
    # 数据格式说明
    st.markdown("""
    <div class="feature-card">
        <h4>📄 What You Need to Provide (27 Basic Variables)</h4>
        <p>The system requires the following 27 basic variables. All other indices will be automatically calculated.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 按类别显示基础变量
    categories = {
        "Demographics (2 variables)": ['age', 'gender'],
        "Lifestyle and Habits (2 variables)": ['smoke', 'exercise'],
        "Physical Examination (11 variables)": ['temperature', 'pulse', 'systolic', 'diastolic', 
                                              'left_naked_eye', 'right_naked_eye', 'heart_rate', 
                                              'hear', 'heart_rhythm', 'height_cm', 'waist_cm', 'weight_kg'],
        "Laboratory Tests (9 variables)": ['HGB', 'PLT', 'Creatinine', 'Urea', 'TC', 'LDL_C', 'HDL_C',
                                          'glucose_mmolL', 'triglycerides_mmolL'],
        "Medical History (3 variables)": ['hypertension_final', 'diabetes_final', 'dyslipidemia_final']
    }
    
    for category, vars_list in categories.items():
        with st.expander(f"📁 {category}", expanded=(category=="Demographics (2 variables)")):
            # 创建该类别的变量表格
            cat_data = []
            for var in vars_list:
                cat_data.append({
                    'Variable Name': var,
                    'Description': feature_descriptions.get(var, '')
                })
            
            cat_df = pd.DataFrame(cat_data)
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
    
    # 自动计算的衍生变量
    st.markdown("""
    <div class="instruction-card">
        <h4>🧮 What the System Automatically Calculates</h4>
        <p>The following indices are automatically calculated from your basic data:</p>
        
        <table style="width:100%; border-collapse: collapse;">
        <tr style="background-color: #3498db; color: white;">
            <th style="padding: 10px; border: 1px solid #dee2e6;">Index</th>
            <th style="padding: 10px; border: 1px solid #dee2e6;">Formula</th>
            <th style="padding: 10px; border: 1px solid #dee2e6;">Description</th>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>count</strong></td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">Hypertension + Diabetes + Dyslipidemia</td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">Number of existing diseases (0-3)</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>RFM</strong></td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">64 - 20×(Height/Waist) + 12×Gender</td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">Relative Fat Mass index</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>TyG</strong></td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">ln[TG(mg/dL) × GLU(mg/dL) / 2]</td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">Triglyceride-glucose index</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>BMI</strong> (intermediate)</td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">Weight(kg) / Height(m)²</td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">Body Mass Index</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>TyG_BMI</strong></td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">TyG × BMI</td>
            <td style="padding: 10px; border: 1px solid #dee2e6;">TyG-BMI composite index</td>
        </tr>
        </table>
        
        <p style="margin-top: 15px;"><strong>Important Notes:</strong></p>
        <ul>
            <li><strong>TyG Calculation:</strong> TG and GLU values in mmol/L are converted to mg/dL using:
                <ul>
                    <li>TG (mg/dL) = TG (mmol/L) × 88.6</li>
                    <li>GLU (mg/dL) = GLU (mmol/L) × 18.0</li>
                </ul>
                Then: TyG = ln[TG(mg/dL) × GLU(mg/dL) / 2]
            </li>
            <li><strong>RFM Calculation:</strong> Gender encoding: 0 = Male, 1 = Female</li>
            <li>All calculations are performed automatically. You only need to provide the basic measurements.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def sidebar_content():
    """
    侧边栏内容
    """
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
        <h3 style="margin: 0;">🏥 CMM-SCREEN</h3>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Community Edition v1.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### ⚠️ Important Notice")
    st.sidebar.warning("""
    • Assessment results are for reference only
    • Cannot replace professional medical diagnosis
    • High-risk individuals should seek medical consultation
    • Only for people aged 65 and above
    • ID column is for record management only, not used in prediction
    """)
    
    # 显示系统信息
    st.sidebar.markdown("---")
    st.sidebar.caption(f"System Version: 1.0 | Date: January 2026")

if __name__ == "__main__":
    main()
    sidebar_content()