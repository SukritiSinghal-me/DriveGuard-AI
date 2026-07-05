import cv2
import streamlit as st

from utils.detector import detect_face
from utils.preprocess import preprocess_face
from utils.predict import predict_drowsiness
from utils.alarm import start_alarm, stop_alarm

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🚗 Driver Drowsiness Detection System")

st.markdown(
    "Real-Time Driver Monitoring using Deep Learning"
)

st.divider()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("Project Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.write("### Model")

st.sidebar.write("Custom CNN")

st.sidebar.write("### Classes")

st.sidebar.write("- Alert")
st.sidebar.write("- Drowsy")

st.sidebar.write("### Input Size")

st.sidebar.write("128 x 128")

st.sidebar.write("### Framework")

st.sidebar.write("TensorFlow + Streamlit")

# ---------------------------------------------------
# Layout
# ---------------------------------------------------

left, right = st.columns([3,1])

with left:

    st.subheader("📷 Live Camera")

    frame_placeholder = st.empty()

with right:

    st.subheader("Driver Status")

    status_box = st.empty()

    st.subheader("Confidence")

    confidence_box = st.empty()

    st.subheader("Fatigue Score")

    fatigue_bar = st.progress(0)

    st.subheader("Alarm")

    alarm_box = st.empty()

st.divider()

# ---------------------------------------------------
# Buttons
# ---------------------------------------------------

col1, col2 = st.columns(2)

start = col1.button("▶ Start Camera")

stop = col2.button("⏹ Stop Camera")

# ---------------------------------------------------
# Camera
# ---------------------------------------------------

if start:

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            st.error("Unable to open webcam.")
            break

        face, coords = detect_face(frame)

        if face is not None:

            processed = preprocess_face(face)

            status, confidence, fatigue = predict_drowsiness(processed)

            x, y, w, h = coords

            if status == "Alert":

                color = (0,255,0)

                stop_alarm()

                status_box.success("🟢 ALERT")

                alarm_box.success("OFF")

            else:

                color = (0,0,255)

                start_alarm()

                status_box.error("🔴 DROWSY")

                alarm_box.error("ON")

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{status} ({confidence:.1f}%)",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            confidence_box.metric(
                "Prediction",
                f"{confidence:.2f}%"
            )

            fatigue_bar.progress(int(fatigue))

        else:

            stop_alarm()

            status_box.warning("No Face")

            confidence_box.metric(
                "Prediction",
                "--"
            )

            fatigue_bar.progress(0)

            alarm_box.info("OFF")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

        if stop:

            break

    cap.release()

    stop_alarm()    