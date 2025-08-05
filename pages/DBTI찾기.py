import os
import streamlit as st
import torch
import numpy as np
import av
from transformers import AutoProcessor, AutoModel
from utils.styles import inject_global_css
from PIL import Image
import pandas as pd
import os

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 페이지 설정
inject_global_css()
logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.png')
st.sidebar.image(logo_path, width=200)

# XCLIP 모델 불러오기
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32").to(device)

# 성향 축 정의 [(코드, 설명)]
axis_prompts = [
    ("C", "이 강아지는 감정적으로 교류하고 신체 접촉을 좋아합니다."),
    ("W", "이 강아지는 반사적으로 본능적으로 행동합니다."),
    ("T", "이 강아지는 신뢰와 안정감을 보입니다."),
    ("N", "이 강아지는 독립적으로 행동하고 필요할 때만 교류합니다."),
    ("E", "이 강아지는 사람과 적극적으로 교류합니다."),
    ("I", "이 강아지는 혼자 있는 것을 좋아합니다."),
    ("A", "이 강아지는 새로운 환경에 호기심이 많고 활발하게 움직입니다."),
    ("L", "이 강아지는 낯선 환경에 적응하지 않고 익숙한 공간을 선호합니다.")
]

nickname_map = {
    "WTIL": "엄마 껌딱지 겁쟁이형", "WTIA": "조심스러운 관찰형",
    "WNIA": "선긋는 외톨이 야생견형", "WNIL": "패닉에 빠진 극소심형",
    "WTEL": "초면엔 신중, 구면엔 친구", "WTEA": "허세 부리는 호기심쟁이",
    "WNEA": "동네 대장 일진형", "WNEL": "까칠한 지킬 앤 하이드형",
    "CTEL": "신이 내린 반려특화형", "CTEA": "인간 사회 적응 만렙형",
    "CNEA": "똥꼬발랄 핵인싸형", "CNEL": "곱게 자란 막내둥이형",
    "CTIA": "가족 빼곤 다 싫어형", "CTIL": "모범견계의 엄친아형",
    "CNIA": "주인에 관심없는 나혼자 산다형", "CNIL": "치고 빠지는 밀당 천재형"
}

# --- DBTI 성향 설명 ---
dbti_descriptions = {
    "C": "감정적으로 교류하며 신체 접촉을 좋아하는 성향",
    "W": "본능적으로 행동하며 반사적인 반응이 많은 성향",
    "T": "안정적이고 사람에 대한 신뢰를 나타내는 성향",
    "N": "자율적으로 행동하고 혼자 있어도 불안하지 않은 성향",
    "E": "사람이나 동물에게 적극적으로 다가가는 성향",
    "I": "혼자 있는 것을 선호하고 조용한 성향",
    "A": "새로운 환경에 호기심이 많고 활발한 성향",
    "L": "익숙한 장소를 선호하고 낯선 곳을 피하는 성향"
}

# 프레임 샘플링 함수
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# 영상 디코딩 함수
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames

# 예측 함수
def predict(video_frames):
    code = ""
    for i in range(0, len(axis_prompts), 2):
        left_code, left_text = axis_prompts[i]
        right_code, right_text = axis_prompts[i+1]

        inputs = processor(
            text=[left_text, right_text],
            videos=video_frames,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_video
            pred = torch.argmax(logits, dim=1).item()
            code += [left_code, right_code][pred]
    return code

# --- MBTI → DBTI 매핑 함수 ---
def mbti_to_dbti(mbti):
    mbti = mbti.upper()
    if len(mbti) != 4:
        return None

    # 각 MBTI 축을 DBTI 코드로 변환
    mapping = {
        0: {"E": "E", "I": "I"},   # 외향/내향
        1: {"S": "L", "N": "A"},   # 감각/직관 → 낯익음/탐색
        2: {"T": "W", "F": "C"},   # 사고/감정 → 본능/감성
        3: {"J": "T", "P": "N"},   # 판단/인식 → 신뢰/자율
    }
    try:
        return mapping[2][mbti[2]] + mapping[3][mbti[3]] + mapping[0][mbti[0]] + mapping[1][mbti[1]]
    except:
        return None

st.title("🐶 당신의 반려견 성향은?")
st.warning("이 기능은 비디오 업로드가 필요합니다. 20초 길이 이상의 mp4 비디오를 업로드해주세요.")

uploaded_video = st.file_uploader("비디오를 업로드하세요 (mp4)", type=["mp4"])

if uploaded_video:
    st.video(uploaded_video)

    # 프레임 추출 단계
    with st.status("프레임 추출 중..."):
        container = av.open(uploaded_video)
        total_frames = container.streams.video[0].frames
        clip_len = min(total_frames, 8)
        indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=1, seg_len=total_frames)
        video_frames = read_video_pyav(container, indices)

    # 예측 단계
    with st.status("성향 분석 중..."):
        code = predict(video_frames)
        nickname = nickname_map.get(code, "알 수 없음")

    st.markdown(f"""
        <div style='background-color:#f4f6f8; padding:1rem; border-radius:10px; margin-top:1rem;'>
        <strong>예측된 DBTI 코드:</strong> <span style='color:#1b64da; font-size:1.2rem;'>{code}</span><br>
        <strong>성향 설명:</strong> <span style='font-weight:500;'>{nickname}</span>
        </div>
        """, unsafe_allow_html=True)

    # MBTI → DBTI 입력 처리
    st.header("🐶 MBTI → DBTI 성향 매칭")

    mbti_input = st.text_input("당신의 MBTI를 입력하세요 (예: INFP)", max_chars=4).upper()
    if mbti_input and len(mbti_input) == 4 and all(c in "EISNTFJP" for c in mbti_input):
        dbti_code = mbti_to_dbti(mbti_input)
        st.success(f"✨ 변환된 DBTI 코드: `{dbti_code}`")
        st.markdown("---")
        st.header("🔍 매칭 결과 보기")
        st.markdown(f"**입력한 MBTI**: `{mbti_input}`")
        st.markdown(f"**변환된 DBTI**: `{dbti_code}`")
        
        st.subheader("📖 DBTI 성향 설명")
        for letter in dbti_code:
            desc = dbti_descriptions.get(letter, "설명 없음")
            st.markdown(f"- **{letter}**: {desc}")

    # 궁합 계산
        st.header("💞 당신과의 궁합은?")
        if code:
            compatibility = sum(a == b for a, b in zip(dbti_code, code)) * 100 / 4
            st.markdown(f"**궁합 점수**: `{compatibility:.1f}%`")
            if compatibility >= 75:
                st.success("👍 찰떡궁합이에요! 함께하면 큰 시너지를 낼 수 있어요.")
            elif compatibility >= 50:
                st.info("😊 괜찮은 궁합이에요. 서로의 차이를 이해하려는 노력이 중요해요.")
            elif compatibility >= 25:
                st.warning("😕 궁합이 조금 아쉬워요. 대화를 통해 다리를 놓아보세요.")
            else:
                st.error("💔 궁합이 낮은 편이에요. 성향 차이를 이해하고 배려가 필요해요.")
        else:
            st.info("👀 궁합 비교를 위해 상대방의 DBTI 정보(code)가 필요합니다.")
