import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from model import build_unet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI–IoT Oil Spill Detection",
    layout="wide",
    page_icon="🛢️"   
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    padding: 1.5rem;
}
.section {
    padding: 1.2rem;
    border-radius: 12px;
    background-color: #0e1117;
    margin-bottom: 1.5rem;
}
h1, h2, h3 {
    color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🛢️ AI-Driven Marine Oil Spill Detection System")
st.caption(
    "Deep Learning–based U-Net segmentation on SAR imagery for early oil spill identification"
)

st.divider()


model.summary()


# ---------------- LOAD MODEL SAFELY ----------------
def load_model_once():
    model = build_unet(input_shape=(256, 256, 1))

    # Build model explicitly
    model.build((None, 256, 256, 1))

    # Load weights safely
    model.load_weights("unet_oil_spill.weights.h5")

    return model


if "model" not in st.session_state:
    with st.spinner("🔄 Loading segmentation model..."):
        st.session_state.model = load_model_once()
    st.success("✅ Model loaded successfully")

model = st.session_state.model

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader(
        "Upload SAR Image",
        type=["png", "jpg", "jpeg"]
    )

    threshold = st.slider(
        "Detection Sensitivity",
        0.1, 0.9, 0.3, 0.05
    )

    st.markdown(
        "ℹ️ *Lower values detect weaker spills but may add noise.*"
    )

# ---------------- MAIN CONTENT ----------------
if uploaded_file is not None:

    # -------- READ IMAGE --------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img_resized = cv2.resize(img, (256, 256))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = img_norm[np.newaxis, ..., np.newaxis]

    # -------- PREDICT --------
    pred = model.predict(img_input, verbose=0)[0]

    pred_mask = (pred > threshold).astype(np.uint8)

    # -------- POST-PROCESSING --------
    kernel = np.ones((3, 3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    min_area = 200
    clean_mask = np.zeros_like(pred_mask)

    contours, _ = cv2.findContours(
        pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 1, thickness=-1)

    pred_mask = clean_mask

    # ---------------- METRICS ----------------
    confidence = float(np.mean(pred))

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Type", "U-Net CNN")
    col2.metric("Avg Spill Probability", f"{confidence:.3f}")
    col3.metric("Detection Status", "Spill Detected" if confidence > 0.1 else "No Spill")

    st.divider()

    # ---------------- VISUALS ----------------
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📡 Input SAR Image")
        st.image(img_resized, clamp=True, caption="Grayscale SAR Input")

    with c2:
        st.subheader("🔥 Oil Spill Probability Heatmap")
        fig, ax = plt.subplots()
        im = ax.imshow(pred.squeeze(), cmap="hot", vmin=0, vmax=1)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ---------------- MASK ----------------
    st.subheader("🟥 Predicted Oil Spill Mask")

    colored_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    colored_mask[pred_mask.squeeze() == 1] = [255, 0, 0]

    st.image(
        colored_mask,
        caption="Red regions indicate detected oil spills",
        use_container_width=True
    )

    # ---------------- OVERLAY ----------------
    st.subheader("🧩 Overlay Visualization")

    fig, ax = plt.subplots()
    ax.imshow(img_resized, cmap="gray")
    ax.imshow(pred.squeeze(), cmap="hot", alpha=0.55)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    # ---------------- INFO ----------------
    if confidence < 0.1:
        st.info("ℹ️ No significant oil spill detected in this SAR image.")

# ----------------- FOOTER ----------------
st.markdown("---")
st.caption("© 2026 | AI–IoT Marine Oil Spill Detection | Sachin Sharma A")

