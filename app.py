import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "model.h5"
WIN_THRESH = 0.695
FILE_THRESH = 0.50
WINDOW = 40
HOP = 20
SR = 16000

# Spectrogram parameters
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FRAME_MS = 100  # Update spectrogram every 100ms

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- AUDIO PROCESSING ----------------
def extract_mel(path):
    y, _ = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel, y

def predict_file(audio_path):
    """Returns overall score and per-window predictions with timestamps"""
    mel, audio = extract_mel(audio_path)
    T = mel.shape[1]
    votes = []
    window_scores = []
    window_times = []
    
    if T < WINDOW:
        mel_padded = np.pad(mel, ((0,0),(0, WINDOW-T)))
        mel_input = mel_padded[..., None][None, ...]
        p = model.predict(mel_input, verbose=0)[0][0]
        votes.append(p > WIN_THRESH)
        window_scores.append(p)
        window_times.append(0)
    else:
        for i in range(0, T - WINDOW + 1, HOP):
            win = mel[:, i:i+WINDOW]
            win = win[..., None][None, ...]
            p = model.predict(win, verbose=0)[0][0]
            votes.append(p > WIN_THRESH)
            window_scores.append(p)
            # Calculate time in seconds for this window
            time_sec = (i + WINDOW/2) * HOP_LENGTH / SR
            window_times.append(time_sec)
    
    ratio = sum(votes) / len(votes)
    return ratio, mel, window_scores, window_times, audio

def create_spectrogram_with_overlay(mel, window_scores, window_times):
    """Create interactive spectrogram with tampering overlay"""
    # Time axis
    duration = mel.shape[1] * HOP_LENGTH / SR
    times = np.linspace(0, duration, mel.shape[1])
    
    # Frequency axis (Mel bins)
    freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=SR/2)
    
    # Create figure
    fig = go.Figure()
    
    # Add spectrogram heatmap
    fig.add_trace(go.Heatmap(
        z=mel,
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar=dict(title="dB"),
        name="Spectrogram",
        hovertemplate='Time: %{x:.2f}s<br>Freq: %{y:.0f}Hz<br>Amplitude: %{z:.1f}dB<extra></extra>'
    ))
    
    # Add tampering overlay regions
    shapes = []
    annotations = []
    
    for i, (score, time_center) in enumerate(zip(window_scores, window_times)):
        if score > WIN_THRESH:
            # Calculate window boundaries
            window_duration = WINDOW * HOP_LENGTH / SR
            time_start = max(0, time_center - window_duration/2)
            time_end = min(duration, time_center + window_duration/2)
            
            # Color based on confidence
            if score > 0.8:
                color = 'rgba(255, 0, 0, 0.4)'  # Red for high confidence
                label = 'High'
            else:
                color = 'rgba(255, 255, 0, 0.3)'  # Yellow for moderate
                label = 'Moderate'
            
            # Add semi-transparent rectangle
            shapes.append(dict(
                type="rect",
                x0=time_start,
                x1=time_end,
                y0=0,
                y1=SR/2,
                fillcolor=color,
                line=dict(width=0),
                layer="above"
            ))
            
            # Add annotation for first occurrence of each type
            if i == 0 or (i > 0 and abs(score - window_scores[i-1]) > 0.15):
                annotations.append(dict(
                    x=time_center,
                    y=SR/2 * 0.9,
                    text=f"{label}<br>{score:.2%}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color.replace('0.4', '0.8').replace('0.3', '0.8'),
                    bgcolor="white",
                    bordercolor=color.replace('0.4', '1').replace('0.3', '1'),
                    borderwidth=2,
                    font=dict(size=10)
                ))
    
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        title="Mel Spectrogram with Tampering Detection Overlay",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        height=600,
        hovermode='closest',
        showlegend=False
    )
    
    return fig

# ---------------- UI ----------------
st.set_page_config(page_title="Audio Tampering Detection", layout="wide")
st.title("🎧 Audio Tampering Detection with Real-Time Visualization")

# Initialize session state
if 'last_file' not in st.session_state:
    st.session_state.last_file = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Create centered column for upload section
col_left, col_center, col_right = st.columns([0.15, 0.7, 0.15])

with col_center:
    uploaded = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

    if uploaded is not None:
        # Check if a new file was uploaded
        current_file_id = uploaded.file_id
        if st.session_state.last_file != current_file_id:
            st.session_state.last_file = current_file_id
            st.session_state.analyzed = False
            st.session_state.results = None
        
        with open("temp.wav", "wb") as f:
            f.write(uploaded.read())

        # Audio player
        st.audio("temp.wav")

        # Dynamic button text
        button_text = "🔍 Analyze Audio" if not st.session_state.analyzed else "🔄 Analyze Again"
        
        if st.button(button_text, use_container_width=True, type="primary"):
            # Show processing state
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Loading audio file...")
            progress_bar.progress(10)
            time.sleep(0.2)
            
            status_text.text("🔄 Computing Mel spectrogram...")
            progress_bar.progress(30)
            
            # Process audio
            score, mel, window_scores, window_times, audio = predict_file("temp.wav")
            st.session_state.analyzed = True
            
            status_text.text("🔄 Running tampering detection on each frame...")
            progress_bar.progress(60)
            time.sleep(0.3)
            
            status_text.text("🔄 Generating visualization with overlay...")
            progress_bar.progress(80)
            
            # Store results
            st.session_state.results = {
                'score': score,
                'mel': mel,
                'window_scores': window_scores,
                'window_times': window_times,
                'audio': audio
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

# Continue with full-width results section below
if uploaded is not None:

    # Display results if available
    if st.session_state.results is not None:
        results = st.session_state.results
        score = results['score']
        mel = results['mel']
        window_scores = results['window_scores']
        window_times = results['window_times']
        
        st.divider()
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Real-Time Spectrogram Analysis")
            
            # Create and display spectrogram
            fig = create_spectrogram_with_overlay(mel, window_scores, window_times)
            st.plotly_chart(fig, use_container_width=True)
            
            # Legend
            st.markdown("""
            **Legend:**
            - 🟦 **Blue-Green-Yellow**: Spectrogram amplitude (dB scale)
            - 🟥 **Red overlay**: High tampering confidence (>80%)
            - 🟨 **Yellow overlay**: Moderate tampering confidence (69.5-80%)
            """)
        
        with col2:
            st.subheader("📈 Detection Results")
            
            # Overall verdict
            if score >= FILE_THRESH:
                st.error("### ⚠️ TAMPERED")
                st.metric("Tampering Confidence", f"{score:.1%}")
                st.caption(f"Threshold: {FILE_THRESH:.0%} | Score exceeds threshold by {(score-FILE_THRESH):.1%}")
            else:
                st.success("### ✅ CLEAN")
                st.metric("Authenticity Score", f"{(1-score):.1%}")
                st.caption(f"Tampering detected in {score:.1%} of windows (below {FILE_THRESH:.0%} threshold)")
            
            st.divider()
            
            # Statistics
            st.markdown("**Analysis Statistics:**")
            tampered_windows = sum(1 for s in window_scores if s > WIN_THRESH)
            total_windows = len(window_scores)
            
            st.metric("Total Windows Analyzed", total_windows)
            st.metric("Tampered Windows", tampered_windows)
            st.metric("Tampering Ratio", f"{score:.1%}")
            
            # Score distribution
            high_conf = sum(1 for s in window_scores if s > 0.8)
            mod_conf = sum(1 for s in window_scores if WIN_THRESH < s <= 0.8)
            
            if tampered_windows > 0:
                st.markdown("**Tampering Breakdown:**")
                st.write(f"🔴 High confidence: {high_conf} windows")
                st.write(f"🟡 Moderate confidence: {mod_conf} windows")
            
            st.divider()
            
            # Explanation
            if score >= FILE_THRESH:
                st.info("""
                **What this means:**  
                The AI detected manipulation signs in this audio. Red/yellow regions on the spectrogram show where tampering was detected.
                
                **Possible tampering types:**
                - 🔪 Audio deletion
                - 🔀 Splicing/insertion
                - ⚡ Speed modifications
                - ✂️ Other editing artifacts
                """)
            else:
                st.info("""
                **What this means:**  
                No significant tampering detected. The audio appears authentic and unedited.
                """)
        
        st.divider()

else:
    st.info("👆 Upload a .wav audio file to begin analysis")
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        ### Instructions:
        1. **Upload** a `.wav` audio file using the file uploader above
        2. **Listen** to the audio using the built-in player
        3. **Click "Analyze Audio"** to start the tampering detection
        4. **View** the real-time Mel spectrogram with tampering overlay
        
        ### Understanding the Visualization:
        - **Spectrogram**: Shows frequency content over time
        - **Red regions**: High confidence tampering detected (>80%)
        - **Yellow regions**: Moderate confidence tampering (69.5-80%)
        - **Hover** over the spectrogram to see detailed values
        
        ### Technical Details:
        - Uses CNN trained on mel-spectrograms
        - Analyzes audio in overlapping windows
        - Detects deletion, splicing, and speed modifications
        """)
