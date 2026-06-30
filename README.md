# Reliable Underwater Smartphone Communication

AI-assisted underwater optical communication system using only a smartphone's flashlight (transmitter) and camera (receiver) — no external hardware. Built as a simulation-based prototype extending ideas from **U-Flash: Improving Underwater Optical Communication by Scattering Effect (IMWUT 2024)**.

The flashlight transmits binary data as ON/OFF light pulses; the camera captures these via the rolling-shutter effect. Underwater scattering, ambient interference, motion, and inter-symbol interference make this link highly unreliable — this project replaces classical signal-processing decoders with CNN and CNN-LSTM models to reduce Bit Error Rate (BER) under these conditions.

## Key Results

| Challenge | Classical Method | ML Result |
|---|---|---|
| Scattering noise (5 difficulty levels) | Fixed threshold fails at medium (BER 0.51); Adaptive threshold fails at extreme (BER 0.45) | **CNN: 0.000 BER** at all five difficulty levels |
| Inter-symbol interference (ISI) | Plain CNN degrades to ~0.24 BER as bleed-through increases | **CNN-LSTM: 0.000 BER** up to bleed=0.3, degrades far more slowly beyond that |
| Motion (drift + jitter) | Plain CNN degrades to ~0.11 BER at severe motion | **CNN-LSTM (motion-augmented): 0.000 BER** at all motion levels |

See `src/outputs/` for the full BER comparison graphs.

## How It Works

1. **Encode** — text is converted to binary (ASCII), then to a flashlight ON/OFF blink sequence
2. **Transmit** — the rolling-shutter camera captures blinks as bright/dark stripes in a single frame
3. **Channel** — underwater scattering simulated via blur, darkening, red-light absorption, noise, ISI bleed-through, and motion
4. **Decode** — a trained CNN (per-frame) or CNN-LSTM (sequence-aware) recovers the bit sequence
5. **Measure** — Bit Error Rate compares recovered vs. original message

## Project Structure

```
src/
├── Signal pipeline       encoder.py, decoder.py, signal_generator.py,
│                         channel_noise.py, threshold_detector.py,
│                         adaptive_threshold.py, ber_calculation.py, main.py
├── Image pipeline        frame_generator.py, camera_noise.py,
│                         flashlight_detector.py
├── CNN (per-frame)       dataset_generator.py, train_cnn_all_levels.py,
│                         cnn_decoder_pipeline.py,
│                         ber_vs_difficulty_experiment.py
├── CNN-LSTM (sequence)   sequence_generator.py, sequence_dataset_generator.py,
│                         train_cnn_lstm.py, ber_cnn_vs_cnn_lstm.py,
│                         ber_vs_motion_experiment.py
├── Demo                  app.py (Streamlit), generate_test_frames.py
└── outputs/               result graphs (.png)
```

## Running the Project

**Install dependencies:**
```bash
pip install tensorflow opencv-python numpy matplotlib streamlit
```

**Run the signal-level demo:**
```bash
cd src
python main.py
```

**Generate datasets and train models** (large — takes time):
```bash
python dataset_generator.py
python train_cnn_all_levels.py
python sequence_dataset_generator.py medium
python train_cnn_lstm.py medium
```

**Run BER experiments:**
```bash
python ber_vs_difficulty_experiment.py
python ber_cnn_vs_cnn_lstm.py
python ber_vs_motion_experiment.py
```

**Launch the interactive demo:**
```bash
streamlit run app.py
```
Opens a browser UI with three tabs: **Sender** (encode & transmit), **Underwater Channel** (visualize degradation effects), and **Receiver** (decode frames with CNN vs CNN-LSTM, including a file-upload mode for testing custom frame sequences).

## Engineering Notes

Two non-trivial bugs were found and fixed during development:

- **BGR/RGB channel mismatch** — OpenCV loads images in BGR order while Keras/PIL trains on RGB; since the underwater simulation specifically suppresses the red channel, this caused a model with 100% training accuracy to produce near-random predictions at inference. Fixed by explicit channel conversion before every prediction call.
- **Darkening math bug** — `cv2.convertScaleAbs` was found to take the absolute value of negative results instead of clipping to zero, paradoxically making near-black pixels brighter than intended. Fixed with proper float-based clipping.

A further finding: a CNN-LSTM trained without motion augmentation performed *worse* than a plain frame-by-frame CNN under motion — a clear demonstration that sequence models are only robust to conditions represented in their training data. Retraining with motion-augmented sequences resolved this, restoring CNN-LSTM's advantage.

## Status

Simulation pipeline, CNN/CNN-LSTM models, three BER experiments, and the Streamlit demo are complete. Not yet implemented: a CNN-based RoI detector (currently classical/contour-based), a wired error-correction layer, and validation against real captured data.

