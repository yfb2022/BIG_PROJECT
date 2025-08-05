import streamlit as st
from utils.styles import inject_global_css
from PIL import Image
import os

st.set_page_config(page_title="DBTI 대시보드", page_icon="🐶")
inject_global_css()

st.title("🐶 반려동물 성향 분석 플랫폼 🐶")

st.markdown("""
### 반가워요! 🐾  
당신의 반려동물은 어떤 성격일까요?

👈 왼쪽 사이드바에서 원하는 기능을 선택해보세요!

- 🧠 **DBTI 분석 보기**: AI로 분석된 반려동물의 성향 결과를 확인해보세요.  
- 💘 **MBTI 궁합 추천**: 사람과 반려동물의 성향 궁합을 알아보고, 찰떡궁합 파트너를 찾아보세요!

> 데이터 기반 성향 분석으로 더 나은 반려생활을 함께 만들어가요! 🐕🐈
""")

logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.png')
st.sidebar.image(logo_path, width=200)