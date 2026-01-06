import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import os





# 1. 저장된 자산(모델, 스케일러, 중요도) 불러오기
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('final_marketing_model.keras')
    scaler = joblib.load('marketing_scaler.pkl')
    importance_df = joblib.load('feature_importance.pkl')
    return model, scaler, importance_df

try:
    model, scaler, importance_df = load_assets()
except Exception as e:
    st.error(f"파일을 로드하는 중 오류가 발생했습니다: {e}")

# 2. 웹 앱 상단 제목 및 성능 지표
st.title("🎯 마케팅 전환 고객 예측 시스템")
st.write("고객 데이터를 입력하면 최적 임계값(0.3568)을 기준으로 전환 가능성을 예측합니다.")

st.markdown("---")
col_acc, col_auc, col_thresh = st.columns(3)
with col_acc:
    st.metric("모델 정확도", "88.5%") 
with col_auc:
    st.metric("모델 AUC", "0.84")       
with col_thresh:
    st.metric("최적 임계값", "0.3568")
st.markdown("---")

# 3. 사이드바: 고객 정보 입력 폼
st.sidebar.header("고객 정보 입력")

def get_user_input():
    # 수치형 변수 12개
    income = st.sidebar.number_input("연간 수입 (Income)", min_value=0, value=50000)
    ad_spend = st.sidebar.number_input("광고 지출 (AdSpend)", min_value=0.0, value=100.0)
    ctr = st.sidebar.slider("클릭률 (CTR)", 0.0, 1.0, 0.05)
    conv_rate = st.sidebar.slider("전환율 (Conv Rate)", 0.0, 1.0, 0.02)
    visits = st.sidebar.number_input("웹사이트 방문 횟수", min_value=0, value=5)
    pages = st.sidebar.number_input("방문당 페이지 수", min_value=0.0, value=2.5)
    time_on_site = st.sidebar.number_input("사이트 체류 시간 (분)", min_value=0.0, value=5.0)
    shares = st.sidebar.number_input("소셜 공유 횟수", min_value=0, value=1)
    e_opens = st.sidebar.number_input("이메일 오픈 횟수", min_value=0, value=2)
    e_clicks = st.sidebar.number_input("이메일 클릭 횟수", min_value=0, value=1)
    purchases = st.sidebar.number_input("과거 구매 횟수", min_value=0, value=0)
    loyalty = st.sidebar.number_input("로열티 포인트", min_value=0, value=100)

    # 범주형 변수 1개
    camp_type = st.sidebar.selectbox("캠페인 유형", ["Awareness", "Consideration", "Conversion", "Retention"])
    
    data = {
        'Income': income, 'AdSpend': ad_spend, 'ClickThroughRate': ctr, 'ConversionRate': conv_rate,
        'WebsiteVisits': visits, 'PagesPerVisit': pages, 'TimeOnSite': time_on_site, 'SocialShares': shares,
        'EmailOpens': e_opens, 'EmailClicks': e_clicks, 'PreviousPurchases': purchases, 'LoyaltyPoints': loyalty,
        'CampaignType_Awareness': 1 if camp_type == "Awareness" else 0,
        'CampaignType_Consideration': 1 if camp_type == "Consideration" else 0,
        'CampaignType_Conversion': 1 if camp_type == "Conversion" else 0,
        'CampaignType_Retention': 1 if camp_type == "Retention" else 0
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# 4. 중앙: 예측 실행 섹션 (넓게 사용)
st.subheader("📊 예측 실행 및 결과")

# 버튼을 누르기 전 가이드 텍스트
if "prediction_done" not in st.session_state:
    st.info("사이드바에 고객 정보를 입력한 후 '결과 확인하기' 버튼을 눌러주세요.")

if st.button("결과 확인하기"):
    st.session_state.prediction_done = True
    
    # 스케일링 대상 수치형 컬럼
    numeric_cols = ['Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 
                    'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 
                    'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints']
    
    # 데이터 전처리
    input_scaled = input_df.copy()
    input_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # 모델 예측
    prob = model.predict(input_scaled, verbose=0)[0][0]
    threshold = 0.3568
    
    # ... (예측 수행 코드 이후) ...

    prob = model.predict(input_scaled, verbose=0)[0][0]
    threshold = 0.3568

    # --- 임계값 마커가 포함된 커스텀 가로 바 생성 ---
    st.write(f"**전환 확률 분석 (임계값: {threshold})**")

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 2))

    # 배경 바
    ax.barh([0], [1.0], color='#F0F2F6', height=0.6)

    # 확률 바 (mako 컬러맵 적용)
    # 0~1 사이의 확률값에 따라 mako 색상 팔레트에서 색상을 가져옴
    cmap = sns.color_palette("viridis", as_cmap=True)
    prob_color = cmap(prob)
    ax.barh([0], [prob], color=prob_color, height=0.6)

    # 3. 임계값 마커 (수직선)
    ax.axvline(x=threshold, color='#31333F', linestyle='--', linewidth=2.5)
    
    # 텍스트 라벨 (한글 적용)
    #ax.text(prob, 0, f' {prob*100:.1f}%', va='center', ha='left', fontsize=12, fontweight='bold', color=prob_color)
    #ax.text(threshold, 0.5, f'임계값 ({threshold})', va='center', ha='center', fontsize=10, color='#31333F', fontweight='bold')
    
    # 디자인 정리
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    sns.despine(left=True, bottom=False)
    st.pyplot(fig)

    if prob >= threshold:
        st.success(f"임계값을 {prob - threshold:.4f} 초과한 유망 고객입니다.")
    else:
        st.warning(f"아쉽습니다. 임계값까지 {threshold - prob:.3f}만큼 부족합니다.")

st.markdown("---")

# 5. 하단: 모델 판단 근거 (전체 너비 활용)
st.subheader("💡 모델의 판단 근거")
st.write("아래 그래프는 전체 모델이 고객을 분류할 때 가장 중요하게 고려하는 변수 순위입니다.")

# 중요도 그래프 시각화 (mako 팔레트 적용)
fig, ax = plt.subplots(figsize=(12, 7)) 
sns.barplot(
    x='importance', 
    y='feature', 
    data=importance_df.sort_values(by='importance', ascending=False), 
    palette='rocket', # 중요도에 따라 색상 자동 배정
    ax=ax
)
# 한글 타이틀 및 라벨
ax.set_title('Feature Importance Analysis Result', fontsize=18, pad=20, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
plt.tight_layout()

st.pyplot(fig)