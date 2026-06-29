"""
app.py - Streamlit Demo
AI-Assisted Underwater Optical Communication System
Run with: streamlit run app.py (from src/ folder)
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
from tensorflow.keras.models import load_model

from encoder import text_to_binary
from decoder import binary_to_text
from sequence_generator import generate_sequence
from camera_noise import add_camera_noise_at_level
from ber_calculation import calculate_ber

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Underwater Optical Comm",
    page_icon="🌊",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        padding: 0.5rem 0;
    }
    .subtitle {
        text-align: center;
        color: #5d6d7e;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .stage-box {
        background: #f0f4f8;
        border-left: 4px solid #2e86c1;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .stage-title {
        font-weight: 600;
        color: #1a5276;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    .bit-on {
        display: inline-block;
        background: #f9e79f;
        border: 1px solid #f1c40f;
        border-radius: 3px;
        padding: 2px 6px;
        margin: 1px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .bit-off {
        display: inline-block;
        background: #2c3e50;
        color: white;
        border-radius: 3px;
        padding: 2px 6px;
        margin: 1px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .arrow {
        text-align: center;
        font-size: 1.5rem;
        color: #2e86c1;
        margin: 0.3rem 0;
    }
    .result-good { color: #1e8449; font-weight: 600; }
    .result-bad  { color: #c0392b; font-weight: 600; }
    .metric-box {
        background: white;
        border: 1px solid #d5d8dc;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.markdown('<div class="main-title">🌊 Underwater Optical Communication System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-assisted flashlight-to-camera data transmission | Based on U-Flash (IMWUT 2024) | Supervised by Prof. Arnab Paul</div>', unsafe_allow_html=True)

# =====================================================
# SESSION STATE - shared between tabs
# =====================================================
if "frames" not in st.session_state:
    st.session_state.frames = None
if "binary" not in st.session_state:
    st.session_state.binary = None
if "message" not in st.session_state:
    st.session_state.message = None
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "medium"
if "motion_strength" not in st.session_state:
    st.session_state.motion_strength = 0.5

# =====================================================
# MODEL LOADER
# =====================================================
@st.cache_resource
def load_cnn(difficulty):
    return load_model(f"flashlight_cnn_{difficulty}.h5")

@st.cache_resource
def load_lstm():
    return load_model("cnn_lstm_medium.h5")

def frame_to_pil(frame, size=80):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return img.resize((size, size), Image.NEAREST)

def decode_cnn(frames, model):
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    batch = np.array(rgb, dtype="float32") / 255.0
    preds = model.predict(batch, verbose=0)
    return "".join('1' if p[0] > 0.5 else '0' for p in preds)

def decode_cnn_lstm(frames, model):
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    seq = np.expand_dims(np.array(rgb, dtype="float32") / 255.0, axis=0)
    preds = model.predict(seq, verbose=0)[0]
    return "".join('1' if p[0] > 0.5 else '0' for p in preds)

# =====================================================
# TABS
# =====================================================
tab_sender, tab_channel, tab_receiver, tab_results = st.tabs([
    "📤 Sender", "🌊 Underwater Channel", "📥 Receiver", "📊 Experiment Results"
])


# =========================================================
# TAB 1: SENDER
# =========================================================
with tab_sender:
    st.header("📤 Sender Pipeline")
    st.markdown("The sending phone encodes a text message into binary, converts it to flashlight ON/OFF pulses, and transmits them through water.")

    col_in, col_pipe = st.columns([1, 2])

    with col_in:
        st.subheader("Message Input")
        message_input = st.text_input("Type your message", value="HI", max_chars=4,
            help="Keep to 1-4 characters for a fast demo")
        difficulty_input = st.selectbox("Scattering difficulty",
            ["easy", "medium", "hard", "extreme", "brutal"], index=1)
        motion_input = st.slider("Motion strength", 0.0, 3.0, 0.5, 0.5)
        bleed_input = st.slider("ISI bleed strength", 0.0, 0.6, 0.15, 0.05)
        send_btn = st.button("📡 Encode & Transmit", use_container_width=True, type="primary")

    with col_pipe:
        if send_btn and message_input.strip():
            msg = message_input.upper().strip()
            binary = text_to_binary(msg)

            # Stage 1: Text -> Binary
            st.markdown('<div class="stage-box"><div class="stage-title">Stage 1 — Text to Binary (ASCII encoding)</div>', unsafe_allow_html=True)
            chars_html = ""
            for char in msg:
                b = text_to_binary(char)
                chars_html += f"<b>{char}</b> = <code>{b}</code>&nbsp;&nbsp;"
            st.markdown(chars_html + "</div>", unsafe_allow_html=True)
            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

            # Stage 2: Binary -> Signal
            st.markdown('<div class="stage-box"><div class="stage-title">Stage 2 — Binary to Flashlight Signal (255=ON, 0=OFF)</div>', unsafe_allow_html=True)
            bits_html = "".join(
                f'<span class="bit-on">1</span>' if b == '1'
                else f'<span class="bit-off">0</span>'
                for b in binary
            )
            st.markdown(bits_html + "</div>", unsafe_allow_html=True)
            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

            # Stage 3: Generate frames
            st.markdown('<div class="stage-box"><div class="stage-title">Stage 3 — Generating raw flashlight frames</div>', unsafe_allow_html=True)

            # generate clean frames (no noise) first to show raw signal
            import sequence_generator as sg
            orig_bleed = sg.BLEED_STRENGTH_BY_DIFFICULTY.get(difficulty_input, 0.15)
            sg.BLEED_STRENGTH_BY_DIFFICULTY[difficulty_input] = bleed_input
            clean_frames = generate_sequence(binary, "easy", motion_strength=0.0)
            sg.BLEED_STRENGTH_BY_DIFFICULTY[difficulty_input] = orig_bleed

            clean_cols = st.columns(min(len(binary), 10))
            for i, (col, bit) in enumerate(zip(clean_cols, binary)):
                col.image(frame_to_pil(clean_frames[i], 56), caption=f"bit={bit}")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

            # Stage 4: Apply underwater noise
            st.markdown('<div class="stage-box"><div class="stage-title">Stage 4 — Applying underwater channel noise (as frames travel through water)</div>', unsafe_allow_html=True)

            sg.BLEED_STRENGTH_BY_DIFFICULTY[difficulty_input] = bleed_input
            noisy_frames = generate_sequence(binary, difficulty_input, motion_strength=motion_input)
            sg.BLEED_STRENGTH_BY_DIFFICULTY[difficulty_input] = orig_bleed

            noisy_cols = st.columns(min(len(binary), 10))
            for i, (col, bit) in enumerate(zip(noisy_cols, binary)):
                col.image(frame_to_pil(noisy_frames[i], 56), caption=f"bit={bit}")
            st.markdown("</div>", unsafe_allow_html=True)

            # save to session state for receiver tab
            st.session_state.frames    = noisy_frames
            st.session_state.binary    = binary
            st.session_state.message   = msg
            st.session_state.difficulty = difficulty_input
            st.session_state.motion_strength = motion_input

            st.success(f"✅ Transmitted {len(binary)} bits ({len(msg)} characters) — go to the **Receiver** tab to decode!")

        elif send_btn:
            st.warning("Please enter a message.")
        else:
            st.info("👈 Enter a message and click **Encode & Transmit** to see the full sender pipeline.")


# =========================================================
# TAB 2: UNDERWATER CHANNEL
# =========================================================
with tab_channel:
    st.header("🌊 Underwater Channel")
    st.markdown("The underwater channel degrades the optical signal through four physical effects:")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. 🔴 Red light absorption**
        Water absorbs red light much faster than blue/green.
        The camera image shifts toward blue-green tones.

        **2. 🌀 Light scattering (blur)**
        Water particles scatter light in all directions,
        spreading the flashlight blob into a larger, softer region.
        Controlled by blur kernel size in `camera_noise.py`.
        """)
    with c2:
        st.markdown("""
        **3. 🌑 Brightness loss (darkening)**
        Overall signal energy is reduced as light is absorbed
        and scattered away from the camera's line of sight.

        **4. 📡 Inter-symbol interference (bleed-through)**
        Scattered light from one bit's flash takes time to
        dissipate, bleeding into the next frame's capture.
        This is what the LSTM specifically addresses.
        """)

    st.divider()
    st.subheader("🔬 Live Effect Visualizer")
    st.markdown("See how each difficulty level transforms a raw flashlight frame:")

    viz_col1, viz_col2 = st.columns([1, 3])
    with viz_col1:
        viz_diff = st.selectbox("Difficulty to visualize",
            ["easy", "medium", "hard", "extreme", "brutal"], index=2,
            key="viz_diff")
        viz_btn = st.button("Show effect", use_container_width=True)

    with viz_col2:
        if viz_btn:
            # generate one clean ON frame and one noisy version side by side
            raw = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.circle(raw, (32, 32), 10, (240, 240, 240), -1)
            noisy = add_camera_noise_at_level(raw.copy(), viz_diff)

            ci1, ci2, ci3 = st.columns(3)
            with ci1:
                st.image(Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)).resize((120,120), Image.NEAREST),
                    caption="Raw (before water)")
            with ci2:
                st.markdown("### →")
            with ci3:
                st.image(Image.fromarray(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)).resize((120,120), Image.NEAREST),
                    caption=f"After '{viz_diff}' channel")

            gray_raw   = cv2.cvtColor(raw,   cv2.COLOR_BGR2GRAY)
            gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
            st.markdown(f"**Max brightness:** {np.max(gray_raw)} → {np.max(gray_noisy)} &nbsp;|&nbsp; "
                        f"**Mean brightness:** {np.mean(gray_raw):.1f} → {np.mean(gray_noisy):.1f}")


# =========================================================
# TAB 3: RECEIVER
# =========================================================
with tab_receiver:
    st.header("📥 Receiver Pipeline")
    st.markdown("The receiving phone's camera captures the degraded frames and uses CNN / CNN-LSTM to recover the original message.")

    # Option A: use frames from sender tab
    # Option B: upload custom frames
    recv_mode = st.radio("Frame source",
        ["Use frames from Sender tab", "Upload my own frames"],
        horizontal=True)

    frames_to_decode = None
    true_binary = None
    true_message = None

    if recv_mode == "Use frames from Sender tab":
        if st.session_state.frames is not None:
            frames_to_decode = st.session_state.frames
            true_binary      = st.session_state.binary
            true_message     = st.session_state.message
            st.success(f"✅ Loaded {len(frames_to_decode)} frames from sender (message: **{true_message}**)")
        else:
            st.warning("No frames found — go to the **Sender** tab first and click Encode & Transmit.")

    else:
        st.markdown("Upload a sequence of frame images (in order, named frame_00.jpg, frame_01.jpg, ...):")
        uploaded = st.file_uploader("Upload frame images", type=["jpg","png"],
            accept_multiple_files=True)
        true_binary_input = st.text_input("True bitstring (optional, for BER calculation)", "")

        if uploaded:
            uploaded_sorted = sorted(uploaded, key=lambda f: f.name)
            frames_to_decode = []
            for f in uploaded_sorted:
                arr = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    frames_to_decode.append(img)
            true_binary = true_binary_input if true_binary_input else None
            st.success(f"✅ Loaded {len(frames_to_decode)} uploaded frames")

    if frames_to_decode is not None:
        decode_diff = st.session_state.difficulty if recv_mode == "Use frames from Sender tab" else "medium"
        decode_btn = st.button("🔍 Decode Frames", use_container_width=True, type="primary")

        if decode_btn:
            # show received frames
            st.subheader("📷 Received Frames (what the camera captured)")
            frame_cols = st.columns(min(len(frames_to_decode), 10))
            for i, (col, frame) in enumerate(zip(frame_cols, frames_to_decode)):
                col.image(frame_to_pil(frame, 56), caption=f"frame {i}")

            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

            # load models
            with st.spinner("Loading models..."):
                cnn_model  = load_cnn(decode_diff)
                lstm_model = load_lstm()

            # decode
            with st.spinner("Decoding..."):
                cnn_bits  = decode_cnn(frames_to_decode,  cnn_model)
                lstm_bits = decode_cnn_lstm(frames_to_decode, lstm_model)

            # results
            st.subheader("🔬 Decoding Results")
            r1, r2 = st.columns(2)

            with r1:
                st.markdown("### 🔴 Plain CNN (frame-by-frame)")
                cnn_msg = binary_to_text(cnn_bits)
                st.markdown(f"**Recovered binary:** `{cnn_bits}`")
                st.markdown(f"**Recovered message:** `{cnn_msg}`")
                if true_binary:
                    _, ber = calculate_ber(true_binary[:len(cnn_bits)], cnn_bits)
                    color = "result-good" if ber == 0 else "result-bad"
                    st.markdown(f'<span class="{color}">BER: {ber:.3f}</span>', unsafe_allow_html=True)
                if true_message:
                    match = cnn_msg == true_message
                    st.markdown("✅ Correct!" if match else "❌ Decoding error")

            with r2:
                st.markdown("### 🟢 CNN-LSTM (sequence-aware)")
                lstm_msg = binary_to_text(lstm_bits)
                st.markdown(f"**Recovered binary:** `{lstm_bits}`")
                st.markdown(f"**Recovered message:** `{lstm_msg}`")
                if true_binary:
                    _, ber = calculate_ber(true_binary[:len(lstm_bits)], lstm_bits)
                    color = "result-good" if ber == 0 else "result-bad"
                    st.markdown(f'<span class="{color}">BER: {ber:.3f}</span>', unsafe_allow_html=True)
                if true_message:
                    match = lstm_msg == true_message
                    st.markdown("✅ Correct!" if match else "❌ Decoding error")

            # bit-by-bit breakdown
            if true_binary:
                st.subheader("🔍 Bit-by-bit Breakdown")
                rows = []
                for i, true_bit in enumerate(true_binary[:len(cnn_bits)]):
                    rows.append({
                        "Bit #": i,
                        "True": true_bit,
                        "CNN prediction": cnn_bits[i],
                        "CNN ✓": "✅" if cnn_bits[i] == true_bit else "❌",
                        "CNN-LSTM prediction": lstm_bits[i],
                        "LSTM ✓": "✅" if lstm_bits[i] == true_bit else "❌",
                    })
                st.dataframe(rows, use_container_width=True, height=250)


# =========================================================
# TAB 4: RESULTS
# =========================================================
with tab_results:
    st.header("📊 Experimental Results")
    st.markdown("Three sets of experiments, each testing a different reliability challenge named in the project brief.")

    res1, res2, res3 = st.tabs([
        "🔊 Scattering Noise", "〰️ Inter-Symbol Interference", "🏃 Motion"
    ])

    with res1:
        st.subheader("BER vs Scattering Difficulty: Fixed vs Adaptive vs CNN")
        st.markdown("""
        **What was tested:** five difficulty levels (easy → brutal) with progressively
        stronger blur, darkening, and noise. A separate CNN was trained per level.

        **Finding:** Fixed threshold collapses to ~50% BER from medium onward
        (random guessing). Adaptive threshold is more robust but fails at extreme/brutal.
        CNN maintains 0% BER across all five levels — recognizing spatial blob patterns
        rather than relying on a single brightness number.
        """)
        try:
            st.image("outputs/ber_vs_difficulty.png", width=600)
        except:
            st.warning("Run plot_ber_results.py to generate this graph.")

    with res2:
        st.subheader("BER vs ISI Bleed Strength: CNN vs CNN-LSTM")
        st.markdown("""
        **What was tested:** inter-symbol interference (light from one bit's flash
        bleeding into the next frame) at increasing bleed strength.

        **Finding:** at bleed=0.2 the plain CNN starts failing (7.3% BER) while
        CNN-LSTM stays perfect. The CNN-LSTM's LSTM layer uses context from
        neighboring frames to distinguish genuine ON bits from residual bleed-through —
        something a frame-by-frame classifier fundamentally cannot do.
        """)
        try:
            st.image("outputs/ber_cnn_vs_cnn_lstm.png", width=600)
        except:
            st.warning("Run ber_cnn_vs_cnn_lstm.py to generate this graph.")

    with res3:
        st.subheader("BER vs Motion Strength: CNN vs CNN-LSTM (motion-augmented)")
        st.markdown("""
        **What was tested:** relative motion between transmitter and receiver
        (blob drifting and jittering across frames) at increasing severity.

        **Key finding:** CNN-LSTM trained *without* motion data actually performed
        *worse* than plain CNN under motion — demonstrating that a model is only
        robust to conditions it was trained on. After retraining with motion-augmented
        data, CNN-LSTM achieves 0% BER across all motion levels while plain CNN
        degrades to ~10% at severe motion.
        """)
        try:
            st.image("outputs/ber_vs_motion.png", width=600)
        except:
            st.warning("Run ber_vs_motion_experiment.py to generate this graph.")

    st.divider()
    st.markdown("""
    <div style='text-align:center; color:gray; font-size:13px;'>
    AI-Assisted Underwater Optical Communication | JSS Academy of Technical Education, Noida |
    C-DAC Internship | Supervised by Prof. Arnab Paul
    </div>
    """, unsafe_allow_html=True)