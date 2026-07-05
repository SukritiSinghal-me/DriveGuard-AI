import av
import cv2
import streamlit as st
from streamlit_webrtc import (
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

from utils.detector import detect_face
from utils.preprocess import preprocess_face
from utils.predict import predict_drowsiness
from utils.alarm import start_alarm, stop_alarm

# ---------------------------------------
# Page Config
# ---------------------------------------

st.set_page_config(
    page_title="DriveGuard AI",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------
# Custom CSS
# ---------------------------------------

st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1565C0;
}

.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
}

.metric-box{
    background:#F5F5F5;
    padding:15px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Title
# ---------------------------------------

st.markdown(
    "<p class='main-title'>🚗 DriveGuard AI</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Real-Time Driver Drowsiness Detection using Deep Learning</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------
# Sidebar
# ---------------------------------------

st.sidebar.title("Project Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("---")

st.sidebar.write("### Model")
st.sidebar.write("Custom CNN")

st.sidebar.write("### Classes")
st.sidebar.write("• Alert")
st.sidebar.write("• Drowsy")

st.sidebar.write("### Input Size")
st.sidebar.write("128 × 128")

st.sidebar.write("### Framework")
st.sidebar.write("TensorFlow")
st.sidebar.write("Streamlit")
st.sidebar.write("OpenCV")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Instructions

1. Click START
2. Allow Camera Permission
3. Sit in front of camera
4. AI predicts Alert/Drowsy
"""
)

# ---------------------------------------
# Layout
# ---------------------------------------

left, right = st.columns([3,1])

with left:

    st.subheader("📷 Live Camera")

    

with right:

    st.subheader("Driver Status")

    status_box = st.empty()

    st.subheader("Confidence")

    confidence_box = st.empty()

    st.subheader("Fatigue Score")

    fatigue_box = st.progress(0)

    st.subheader("Alarm")

    alarm_box = st.empty()

st.divider()

# ---------------------------------------
# WebRTC Configuration
# ---------------------------------------

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)


# ==========================================================
# Session State
# ==========================================================

if "status" not in st.session_state:
    st.session_state.status = "Waiting..."

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "fatigue" not in st.session_state:
    st.session_state.fatigue = 0

if "alarm" not in st.session_state:
    st.session_state.alarm = False

if "face_detected" not in st.session_state:
    st.session_state.face_detected = False


# ==========================================================
# Video Processor
# ==========================================================

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        face, coords = detect_face(image)

        if face is not None:

            st.session_state.face_detected = True

            face = preprocess_face(face)

            status, confidence, fatigue = predict_drowsiness(face)

            st.session_state.status = status
            st.session_state.confidence = confidence
            st.session_state.fatigue = fatigue

            if status == "Drowsy":

                st.session_state.alarm = True
                start_alarm()

                color = (0,0,255)

            else:

                st.session_state.alarm = False
                stop_alarm()

                color = (0,255,0)

            x, y, w, h = coords

            cv2.rectangle(
                image,
                (x,y),
                (x+w,y+h),
                color,
                2
            )

            cv2.putText(
                image,
                status,
                (x,y-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                image,
                f"{confidence:.1f}%",
                (x,y+h+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        else:

            stop_alarm()

            st.session_state.face_detected = False
            st.session_state.status = "No Face"
            st.session_state.confidence = 0
            st.session_state.fatigue = 0
            st.session_state.alarm = False

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )



# ==========================================================
# Start Webcam
# ==========================================================

webrtc_ctx = webrtc_streamer(
    key="driver-drowsiness",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_processor_factory=VideoProcessor,
    async_processing=True,
)



# ==========================================================
# Live Dashboard
# ==========================================================

status = st.session_state.status
confidence = st.session_state.confidence
fatigue = st.session_state.fatigue
alarm = st.session_state.alarm

if status == "Alert":

    status_box.success("🟢 ALERT")

elif status == "Drowsy":

    status_box.error("🔴 DROWSY")

else:

    status_box.warning("🟡 No Face")


confidence_box.metric(
    "Confidence",
    f"{confidence:.2f}%"
)

fatigue_box.progress(min(max(int(fatigue), 0), 100))

if alarm:

    alarm_box.error("🚨 Alarm ON")

else:

    alarm_box.success("✅ Alarm OFF")




# ==========================================================
# Footer
# ==========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        🚗 <b>DriveGuard AI</b><br><br>

        Real-Time Driver Drowsiness Detection using
        TensorFlow • OpenCV • Streamlit • WebRTC
    </div>
    """,
    unsafe_allow_html=True,
)
