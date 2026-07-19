"""
app.py - Streamlit Demo
AI-Assisted Underwater Optical Communication System
Run with: streamlit run app.py (from src/ folder)

Architecture note:
  Sender    -> encodes text into binary and generates the ideal flashlight
               pulse sequence. The sender has no knowledge of the water.
  Channel   -> the physical underwater medium. Scattering, motion and
               inter-symbol-interference (ISI) bleed are properties of the
               CHANNEL, not the sender, and are configured here.
  Receiver  -> captures the degraded frames and decodes them with CNN /
               CNN-LSTM models.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from tensorflow.keras.models import load_model

# Ensure all relative paths (models, outputs/) resolve correctly
# regardless of the working directory the app is launched from
APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)

from encoder import text_to_binary
from decoder import binary_to_text
from sequence_generator import generate_sequence
import sequence_generator as sg
from camera_noise import add_camera_noise_at_level
from ber_calculation import calculate_ber

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Underwater Optical Comm",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================
# CUSTOM CSS — dark, modern, "deep water" theme
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg:        #070b12;
        --panel:     #0e1420;
        --panel-2:   #121a29;
        --border:    #1e2a3d;
        --cyan:      #22d3ee;
        --cyan-dim:  rgba(34, 211, 238, 0.12);
        --coral:     #fb7550;
        --coral-dim: rgba(251, 117, 80, 0.14);
        --text:      #e8edf5;
        --text-dim:  #8a97ab;
        --good:      #34d399;
        --bad:       #fb7185;
    }

    .stApp { background: var(--bg); }
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; color: var(--text); }

    /* ---------- Hero header ---------- */
    .hero {
        text-align: center;
        padding: 1.4rem 1rem 1.8rem 1rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #22d3ee 0%, #7dd3fc 45%, #fb7550 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.35rem;
    }
    .hero-sub {
        color: var(--text-dim);
        font-size: 0.95rem;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: var(--cyan-dim);
        color: var(--cyan);
        border: 1px solid rgba(34,211,238,0.3);
        border-radius: 999px;
        padding: 0.15rem 0.75rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* ---------- Pipeline stage cards ---------- */
    .stage-box {
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 3px solid var(--cyan);
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin: 0.6rem 0;
    }
    .stage-box.channel { border-left-color: var(--coral); }
    .stage-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: var(--text);
        font-size: 0.95rem;
        margin-bottom: 0.55rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .stage-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px; height: 22px;
        border-radius: 50%;
        background: var(--cyan-dim);
        color: var(--cyan);
        font-size: 0.75rem;
        font-weight: 700;
    }
    .stage-box.channel .stage-num { background: var(--coral-dim); color: var(--coral); }

    .arrow {
        text-align: center;
        font-size: 1.3rem;
        color: var(--text-dim);
        margin: 0.1rem 0;
        line-height: 1;
    }

    /* ---------- bit chips ---------- */
    .bit-on, .bit-off {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 5px;
        padding: 3px 7px;
        margin: 2px;
    }
    .bit-on  { background: var(--cyan-dim); color: var(--cyan); border: 1px solid rgba(34,211,238,0.35); }
    .bit-off { background: #131a26; color: var(--text-dim); border: 1px solid var(--border); }

    .char-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: var(--text);
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 3px 9px;
        margin: 2px 6px 2px 0;
        display: inline-block;
    }
    .char-chip b { color: var(--coral); }

    /* ---------- result text ---------- */
    .result-good { color: var(--good); font-weight: 600; }
    .result-bad  { color: var(--bad);  font-weight: 600; }

    /* ---------- info banner (idle state) ---------- */
    .info-banner {
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: var(--text-dim);
        font-size: 0.9rem;
    }

    /* ---------- section labels ---------- */
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--text);
        margin: 0.3rem 0 0.6rem 0;
    }

    /* ---------- streamlit element overrides ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-dim);
        font-weight: 500;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--cyan) !important;
        background: var(--cyan-dim) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, var(--coral), #ff8f6b);
        border: none;
        font-weight: 600;
        border-radius: 8px;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: var(--panel-2);
        border: 1px solid var(--border);
        color: var(--text);
        border-radius: 8px;
    }

    footer.app-footer {
        text-align: center;
        color: var(--text-dim);
        font-size: 0.78rem;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid var(--border);
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">Sender → Channel → Receiver</div>
    <div class="hero-title">🌊 Underwater Optical Communication System</div>
    <div class="hero-sub">AI-assisted flashlight-to-camera data transmission</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
defaults = {
    "message": None,
    "binary": None,
    "clean_frames": None,     # ideal pulses generated by the sender
    "noisy_frames": None,     # frames after passing through the channel
    "difficulty": "medium",
    "motion_strength": 0.5,
    "bleed_strength": 0.15,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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

def stage_header(num, title, channel=False):
    cls = "stage-box channel" if channel else "stage-box"
    st.markdown(
        f'<div class="{cls}"><div class="stage-title">'
        f'<span class="stage-num">{num}</span>{title}</div>',
        unsafe_allow_html=True
    )

# =====================================================
# TABS
# =====================================================
tab_sender, tab_channel, tab_receiver, tab_results = st.tabs([
    "📤  Sender", "🌊  Underwater Channel", "📥  Receiver", "📊  Experiment Results"
])


# =========================================================
# TAB 1: SENDER — encoding only, no water physics
# =========================================================
with tab_sender:
    st.header("📤 Sender Pipeline")
    st.markdown(
        "The sending phone only does two things: turn text into binary, and turn binary into a "
        "flashlight ON/OFF pulse sequence. **The sender has no knowledge of, or control over, "
        "the water** — scattering, motion and bleed are channel effects, configured in the "
        "*Underwater Channel* tab."
    )

    col_in, col_pipe = st.columns([1, 2])

    with col_in:
        st.markdown('<div class="section-label">Message Input</div>', unsafe_allow_html=True)
        message_input = st.text_input("Type your message", value="HI", max_chars=4,
            help="Keep to 1-4 characters for a fast demo")
        send_btn = st.button("📡 Encode & Transmit", use_container_width=True, type="primary")
        st.caption("Scattering / motion / ISI sliders live in the **Underwater Channel** tab — "
                   "they describe the water, not the transmitter.")

    with col_pipe:
        if send_btn and message_input.strip():
            msg = message_input.upper().strip()
            binary = text_to_binary(msg)

            # Stage 1: Text -> Binary
            stage_header(1, "Text → Binary (ASCII encoding)")
            chars_html = ""
            for char in msg:
                b = text_to_binary(char)
                chars_html += f'<span class="char-chip"><b>{char}</b> = {b}</span>'
            st.markdown(chars_html + "</div>", unsafe_allow_html=True)
            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

            # Stage 2: Binary -> Signal
            stage_header(2, "Binary → Flashlight Signal (255 = ON, 0 = OFF)")
            bits_html = "".join(
                f'<span class="bit-on">1</span>' if b == '1'
                else f'<span class="bit-off">0</span>'
                for b in binary
            )
            st.markdown(bits_html + "</div>", unsafe_allow_html=True)
            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

            # Stage 3: Generate ideal (clean) pulse frames — no channel applied
            stage_header(3, "Ideal transmitted pulses (before entering the water)")
            clean_frames = generate_sequence(binary, "easy", motion_strength=0.0)

            clean_cols = st.columns(min(len(binary), 10))
            for i, (col, bit) in enumerate(zip(clean_cols, binary)):
                col.image(frame_to_pil(clean_frames[i], 56), caption=f"bit={bit}")
            st.markdown("</div>", unsafe_allow_html=True)

            # save to session state — this is the signal that ENTERS the channel
            st.session_state.message      = msg
            st.session_state.binary       = binary
            st.session_state.clean_frames = clean_frames
            st.session_state.noisy_frames = None  # invalidate old channel run

            st.success(f"✅ Encoded {len(binary)} bits ({len(msg)} characters) — head to the "
                       f"**Underwater Channel** tab to simulate transmission through water.")

        elif send_btn:
            st.warning("Please enter a message.")
        else:
            st.markdown(
                '<div class="info-banner">👈 Enter a message and click <b>Encode & Transmit</b> '
                'to see the sender-side encoding pipeline.</div>',
                unsafe_allow_html=True
            )


# =========================================================
# TAB 2: UNDERWATER CHANNEL — scattering / motion / ISI live here
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
        """)
    with c2:
        st.markdown("""
        **3. 🌑 Brightness loss (darkening)**
        Overall signal energy is reduced as light is absorbed
        and scattered away from the camera's line of sight.

        **4. 📡 Inter-symbol interference (bleed-through)**
        Scattered light from one bit's flash takes time to
        dissipate, bleeding into the next frame's capture.
        """)

    st.divider()

    # ---------------- Channel configuration + simulation on sender's signal ----------------
    st.markdown('<div class="section-label">Configure & Simulate the Channel</div>', unsafe_allow_html=True)
    st.markdown(
        "These parameters describe the **water conditions**, not the transmitter — "
        "turbidity/particle density (scattering), relative sender-receiver motion, and how much "
        "light bleeds between consecutive frames."
    )

    cfg1, cfg2, cfg3, cfg4 = st.columns([1, 1, 1, 0.7])
    with cfg1:
        difficulty_input = st.selectbox("Scattering difficulty",
            ["easy", "medium", "hard", "extreme", "brutal"],
            index=["easy", "medium", "hard", "extreme", "brutal"].index(st.session_state.difficulty))
    with cfg2:
        motion_input = st.slider("Motion strength", 0.0, 3.0, st.session_state.motion_strength, 0.5)
    with cfg3:
        bleed_input = st.slider("ISI bleed strength", 0.0, 0.6, st.session_state.bleed_strength, 0.05)
    with cfg4:
        st.write("")
        st.write("")
        apply_btn = st.button("🌊 Apply Channel", use_container_width=True, type="primary")

    if apply_btn:
        if st.session_state.binary is None:
            st.markdown(
                '<div class="info-banner">⚠️ No transmitted signal yet — go to the '
                '<b>Sender</b> tab and click Encode &amp; Transmit first, then come back '
                'and click Apply Channel.</div>',
                unsafe_allow_html=True
            )
        else:
            orig_bleed = sg.BLEED_STRENGTH_BY_DIFFICULTY.get(difficulty_input, 0.15)
            sg.BLEED_STRENGTH_BY_DIFFICULTY[difficulty_input] = bleed_input
            noisy_frames = generate_sequence(
                st.session_state.binary, difficulty_input, motion_strength=motion_input
            )
            sg.BLEED_STRENGTH_BY_DIFFICULTY[difficulty_input] = orig_bleed

            st.session_state.noisy_frames    = noisy_frames
            st.session_state.difficulty      = difficulty_input
            st.session_state.motion_strength = motion_input
            st.session_state.bleed_strength  = bleed_input

    if st.session_state.binary is None and not apply_btn:
        st.caption("💡 Tip: encode a message in the **Sender** tab first, then apply the channel here.")

    if st.session_state.noisy_frames is not None:
        binary = st.session_state.binary
        stage_header(1, "Signal entering the water (ideal pulses)", channel=True)
        in_cols = st.columns(min(len(binary), 10))
        for i, (col, bit) in enumerate(zip(in_cols, binary)):
            col.image(frame_to_pil(st.session_state.clean_frames[i], 56), caption=f"bit={bit}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="arrow">↓ through water ↓</div>', unsafe_allow_html=True)

        stage_header(2, f"Signal after the channel ({st.session_state.difficulty}, "
                        f"motion={st.session_state.motion_strength}, "
                        f"bleed={st.session_state.bleed_strength})", channel=True)
        out_cols = st.columns(min(len(binary), 10))
        for i, (col, bit) in enumerate(zip(out_cols, binary)):
            col.image(frame_to_pil(st.session_state.noisy_frames[i], 56), caption=f"bit={bit}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.success("✅ Channel simulation applied — go to the **Receiver** tab to decode.")

    st.divider()
    st.markdown('<div class="section-label">🔬 Live Effect Visualizer</div>', unsafe_allow_html=True)
    st.markdown("See how each difficulty level transforms a single raw flashlight frame in isolation:")

    viz_col1, viz_col2 = st.columns([1, 3])
    with viz_col1:
        viz_diff = st.selectbox("Difficulty to visualize",
            ["easy", "medium", "hard", "extreme", "brutal"], index=2,
            key="viz_diff")
        viz_btn = st.button("Show effect", use_container_width=True)

    with viz_col2:
        if viz_btn:
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

    recv_mode = st.radio("Frame source",
        ["Use frames from Underwater Channel tab", "Upload my own frames"],
        horizontal=True)

    frames_to_decode = None
    true_binary = None
    true_message = None

    if recv_mode == "Use frames from Underwater Channel tab":
        if st.session_state.noisy_frames is not None:
            frames_to_decode = st.session_state.noisy_frames
            true_binary      = st.session_state.binary
            true_message     = st.session_state.message
            st.success(f"✅ Loaded {len(frames_to_decode)} channel-degraded frames "
                       f"(message: **{true_message}**)")
        else:
            st.markdown(
                '<div class="info-banner">⚠️ No channel-simulated frames found — encode a '
                'message in the <b>Sender</b> tab, then apply a channel in the '
                '<b>Underwater Channel</b> tab.</div>',
                unsafe_allow_html=True
            )

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
        decode_diff = st.session_state.difficulty if recv_mode == "Use frames from Underwater Channel tab" else "medium"
        decode_btn = st.button("🔍 Decode Frames", use_container_width=True, type="primary")

        if decode_btn:
            st.markdown('<div class="section-label">📷 Received Frames (what the camera captured)</div>', unsafe_allow_html=True)
            frame_cols = st.columns(min(len(frames_to_decode), 10))
            for i, (col, frame) in enumerate(zip(frame_cols, frames_to_decode)):
                col.image(frame_to_pil(frame, 56), caption=f"frame {i}")

            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

            with st.spinner("Loading models..."):
                cnn_model  = load_cnn(decode_diff)
                lstm_model = load_lstm()

            with st.spinner("Decoding..."):
                cnn_bits  = decode_cnn(frames_to_decode,  cnn_model)
                lstm_bits = decode_cnn_lstm(frames_to_decode, lstm_model)

            st.markdown('<div class="section-label">🔬 Decoding Results</div>', unsafe_allow_html=True)
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

            if true_binary:
                st.markdown('<div class="section-label">🔍 Bit-by-bit Breakdown</div>', unsafe_allow_html=True)
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
        except Exception:
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
        except Exception:
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
        except Exception:
            st.warning("Run ber_vs_motion_experiment.py to generate this graph.")

st.markdown("""
<footer class="app-footer">
    AI-Assisted Underwater Optical Communication ·
</footer>
""", unsafe_allow_html=True)