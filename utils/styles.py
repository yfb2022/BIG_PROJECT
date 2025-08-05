import streamlit as st

def inject_global_css():
    st.markdown("""
        <style>
        /* ✅ 사이드바: 파란 배경, 흰색 텍스트 */
        [data-testid="stSidebar"] {
            background-color: #3182f6 !important;
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] * {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* ✅ 버튼 스타일: Toss 블루 */
        button[kind="primary"] {
            background-color: #3182f6 !important;
            color: white !important;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            border: none;
        }

        button[kind="primary"]:hover {
            background-color: #1b64da !important;
        }

        /* ✅ 제목 및 본문 텍스트 */
        h1, h2, h3, h4 {
            color: #1f2d3d !important;
        }

        /* ✅ 링크 컬러 */
        a {
            color: #3182f6 !important;
        }
        </style>
    """, unsafe_allow_html=True)

