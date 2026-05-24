
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import io

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# =========================
# Basic setup
# =========================

st.set_page_config(
    page_title="AI Welgrade by PBS",
    page_icon="🧠",
    layout="wide"
)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "mig_welding_lite_model.keras"
CLASS_NAMES_PATH = APP_DIR / "class_names.json"
FEEDBACK_CSV = APP_DIR / "user_evaluation_cqi.csv"

IMG_SIZE = (224, 224)


# =========================
# Custom CSS
# =========================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .header-card {
        background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
        border: 1px solid #dbe7ff;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10233f;
        margin-bottom: 6px;
    }

    .app-description {
        font-size: 1.05rem;
        color: #42526e;
        line-height: 1.55;
    }

    .white-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e6ebf2;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }

    .good-card {
        border-left: 8px solid #16a34a;
        background-color: #f0fdf4;
        padding: 16px;
        border-radius: 12px;
    }

    .moderate-card {
        border-left: 8px solid #f59e0b;
        background-color: #fffbeb;
        padding: 16px;
        border-radius: 12px;
    }

    .bad-card {
        border-left: 8px solid #dc2626;
        background-color: #fef2f2;
        padding: 16px;
        border-radius: 12px;
    }

    .limit-card {
        border-left: 8px solid #334155;
        background-color: #f1f5f9;
        padding: 18px;
        border-radius: 12px;
        color: #1e293b;
    }

    .footer-note {
        color: #64748b;
        font-size: 0.85rem;
        text-align: center;
        padding-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Load model
# =========================

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model_and_classes()


# =========================
# Helper functions
# =========================

def display_label(label):
    mapping = {
        "good_weld": "Good Weld",
        "moderate_weld": "Moderate Weld",
        "bad_weld": "Bad Weld"
    }
    return mapping.get(label, label)


def display_icon(label):
    mapping = {
        "good_weld": "✅",
        "moderate_weld": "⚠️",
        "bad_weld": "❌"
    }
    return mapping.get(label, "🔎")


def display_color(label):
    mapping = {
        "good_weld": "#16a34a",
        "moderate_weld": "#f59e0b",
        "bad_weld": "#dc2626"
    }
    return mapping.get(label, "#334155")


def confidence_status(confidence):
    if confidence >= 80:
        return "High Confidence", "The AI prediction is relatively confident."
    elif confidence >= 60:
        return "Moderate Confidence", "The AI prediction is usable but instructor review is recommended."
    else:
        return "Low Confidence", "Instructor review is strongly required before using this result."


def get_feedback(predicted_class):
    data = {
        "good_weld": {
            "summary": "The weld is visually classified as Good Weld.",
            "feedback": "The weld bead appears visually acceptable with relatively consistent bead shape, smooth appearance and minimal visible surface defects.",
            "improvement": "Maintain consistent travel speed, torch angle and arc control."
        },
        "moderate_weld": {
            "summary": "The weld is visually classified as Moderate Weld.",
            "feedback": "The weld appears visually acceptable but may show minor issues such as slight bead inconsistency, waviness, minor spatter or surface irregularity.",
            "improvement": "Improve travel speed consistency, torch angle stability and bead formation control."
        },
        "bad_weld": {
            "summary": "The weld is visually classified as Bad Weld.",
            "feedback": "The weld may contain obvious visual defects or poor bead formation. Possible issues may include severe spatter, visible porosity, undercut, overlap, burn-through, poor alignment or highly inconsistent bead appearance.",
            "improvement": "Repeat welding practice and focus on travel speed, torch angle, arc length and bead consistency."
        }
    }
    return data.get(predicted_class, data["moderate_weld"])


def predict_image(image):
    image = image.convert("RGB")
    resized = image.resize(IMG_SIZE)
    arr = np.array(resized)
    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index] * 100)

    probabilities = {
        cls: float(prob * 100) for cls, prob in zip(class_names, prediction)
    }

    return predicted_class, confidence, probabilities


def make_after_image(image, predicted_class, confidence):
    image = image.convert("RGB")
    color = display_color(predicted_class)

    border = 18
    top_bar = 70
    w, h = image.size

    canvas = Image.new("RGB", (w + border * 2, h + border * 2 + top_bar), color)
    draw = ImageDraw.Draw(canvas)

    label_text = f"AI Classification: {display_label(predicted_class)} | Confidence: {confidence:.2f}%"
    draw.text((20, 24), label_text, fill="white")

    canvas.paste(image, (border, top_bar + border))
    return canvas


def rubric_scores(predicted_class, confidence, project_type):
    if predicted_class == "good_weld":
        base = {
            "Complete Weld": 5,
            "Defects": 4,
            "Weld Waviness": 4,
            "Travel Speed": 4,
            "Misalignment": 4
        }
        if confidence >= 85:
            base["Defects"] = 5
            base["Weld Waviness"] = 5
            base["Travel Speed"] = 5

    elif predicted_class == "moderate_weld":
        base = {
            "Complete Weld": 4,
            "Defects": 3,
            "Weld Waviness": 3,
            "Travel Speed": 3,
            "Misalignment": 3
        }
        if confidence < 65:
            base["Defects"] = 2
            base["Weld Waviness"] = 2

    else:
        base = {
            "Complete Weld": 2,
            "Defects": 1,
            "Weld Waviness": 2,
            "Travel Speed": 2,
            "Misalignment": 2
        }
        if confidence >= 80:
            base["Travel Speed"] = 1

    if project_type == "Project 1 - Straight Bead / Start-Stop":
        criteria = ["Complete Weld", "Defects", "Weld Waviness", "Travel Speed"]
        max_mark = 20
    else:
        criteria = ["Complete Weld", "Defects", "Weld Waviness", "Travel Speed", "Misalignment"]
        max_mark = 25

    selected = {c: base[c] for c in criteria}
    total = sum(selected.values())
    percentage = (total / max_mark) * 100

    return selected, total, max_mark, percentage


def create_pdf_report(result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#10233f"),
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#10233f"),
        spaceBefore=10,
        spaceAfter=8
    )

    story = []
    story.append(Paragraph("AI Welgrade by PBS", title_style))
    story.append(Paragraph("AI-Assisted Visual Welding Assessment Report", styles["Heading2"]))
    story.append(Spacer(1, 10))

    info_table = Table([
        ["Student / User Name", result.get("student_name", "-")],
        ["Company / Class", result.get("company", "-")],
        ["Welding Process", result.get("welding_process", "-")],
        ["Project Type", result.get("project_type", "-")],
        ["Report Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ], colWidths=[160, 330])

    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4ff")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7e2")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(info_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI Classification Result", heading_style))

    result_table = Table([
        ["Predicted Class", display_label(result["predicted_class"])],
        ["Confidence Score", f'{result["confidence"]:.2f}%'],
        ["Confidence Status", result["confidence_status"]],
        ["Assessment Status", "AI-assisted preliminary assessment. Instructor review required."]
    ], colWidths=[160, 330])

    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7e2")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(result_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Rubric Score", heading_style))

    rows = [["Criteria", "Score"]]
    for criterion, score in result["scores"].items():
        rows.append([criterion, f"{score} / 5"])
    rows.append(["Total Score", f'{result["total_score"]} / {result["max_mark"]}'])
    rows.append(["Percentage", f'{result["percentage"]:.2f}%'])

    rubric_table = Table(rows, colWidths=[300, 190])
    rubric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#10233f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7e2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, -2), (-1, -1), colors.HexColor("#eef4ff")),
        ("FONTNAME", (0, -2), (-1, -1), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(rubric_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Feedback", heading_style))
    story.append(Paragraph(result["feedback"]["summary"], styles["BodyText"]))
    story.append(Paragraph(result["feedback"]["feedback"], styles["BodyText"]))
    story.append(Paragraph("<b>Recommended Improvement:</b> " + result["feedback"]["improvement"], styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Important Limitation", heading_style))
    limitation = (
        "This system performs external visual image classification only. It does not assess internal penetration, "
        "internal fusion, metallurgical properties or actual weld strength. The generated score is an AI-assisted "
        "preliminary score and must not replace instructor judgement, physical measurement, destructive testing or "
        "non-destructive testing."
    )
    story.append(Paragraph(limitation, styles["BodyText"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


def save_cqi(row):
    new_df = pd.DataFrame([row])

    if FEEDBACK_CSV.exists():
        old_df = pd.read_csv(FEEDBACK_CSV)
        all_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        all_df = new_df

    all_df.to_csv(FEEDBACK_CSV, index=False)
    return all_df




def render_header():
    logo_candidates = [
        APP_DIR / "pbs_logo.png",
        APP_DIR / "logo_pbs.png",
        APP_DIR / "logo.png"
    ]

    logo_path = None
    for path in logo_candidates:
        if path.exists():
            logo_path = path
            break

    st.markdown('<div class="header-card">', unsafe_allow_html=True)

    col_logo, col_title = st.columns([1, 6])

    with col_logo:
        if logo_path:
            st.image(str(logo_path), width=120)
        else:
            st.markdown("### 🧠")

    with col_title:
        st.markdown('<div class="app-title">AI Welgrade by PBS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="app-description">'
            'An AI-based visual welding assessment tool designed to analyse uploaded or captured welding images '
            'and provide visual quality classification, preliminary rubric scoring and feedback for teaching and learning support.'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Navigation
# =========================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Welding Assessment",
        "Result & Rubric Score",
        "PDF Report",
        "User Evaluation / CQI Feedback",
        "About & Limitation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("AI Welgrade by PBS | Lite Mode")

render_header()


# =========================
# Home
# =========================

if page == "Home":
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.header("Welcome")

    st.write(
        "AI Welgrade by PBS is a web-based AI-assisted welding assessment application. "
        "It analyses welding images and provides visual quality classification, preliminary rubric scoring and feedback."
    )

    st.write(
        "This Lite Mode version is designed to support teaching and learning activities. "
        "It is not intended to replace instructor judgement or formal welding inspection."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🧠 **AI Classification**\n\nGood Weld, Moderate Weld or Bad Weld.")

    with col2:
        st.info("📋 **Rubric Scoring**\n\nPreliminary score based on selected project type.")

    with col3:
        st.info("📄 **PDF Report**\n\nDownloadable report for documentation.")


# =========================
# Welding Assessment
# =========================

elif page == "Welding Assessment":
    st.header("Welding Assessment")

    col_a, col_b = st.columns(2)

    with col_a:
        student_name = st.text_input("Student / User Name")
        company = st.text_input("Company / Class")

    with col_b:
        welding_process = st.text_input("Welding Process")
        project_type = st.radio(
            "Select Welding Project Type",
            [
                "Project 1 - Straight Bead / Start-Stop",
                "Project 2 - Butt Joint"
            ]
        )

    input_method = st.radio(
        "Choose Image Input Method",
        ["Upload Image", "Capture Image"],
        horizontal=True
    )

    if input_method == "Upload Image":
        image_file = st.file_uploader(
            "Upload welding image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear welding image with proper lighting and visible weld bead."
        )
    else:
        image_file = st.camera_input("Capture welding image")

    if st.button("Run AI Analysis", type="primary"):
        if image_file is None:
            st.error("Please upload or capture a welding image first.")
        else:
            image = Image.open(image_file).convert("RGB")

            predicted_class, confidence, probabilities = predict_image(image)
            scores, total_score, max_mark, percentage = rubric_scores(predicted_class, confidence, project_type)
            status, status_message = confidence_status(confidence)
            after_image = make_after_image(image, predicted_class, confidence)

            result = {
                "student_name": student_name,
                "company": company,
                "welding_process": welding_process,
                "project_type": project_type,
                "image": image,
                "after_image": after_image,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "confidence_status": status,
                "confidence_message": status_message,
                "scores": scores,
                "total_score": total_score,
                "max_mark": max_mark,
                "percentage": percentage,
                "feedback": get_feedback(predicted_class),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            st.session_state["assessment_result"] = result

            st.success("AI analysis completed. Go to Result & Rubric Score page to view the full result.")

            st.subheader("Quick Result")
            st.write(f"**Predicted Class:** {display_icon(predicted_class)} {display_label(predicted_class)}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.write(f"**Preliminary Score:** {total_score} / {max_mark}")


# =========================
# Result & Rubric Score
# =========================

elif page == "Result & Rubric Score":
    st.header("Result & Rubric Score")

    result = st.session_state.get("assessment_result")

    if result is None:
        st.warning("No assessment result available. Please run analysis on the Welding Assessment page first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Before Analysis")
            st.image(result["image"], use_container_width=True)

        with col2:
            st.subheader("After Analysis")
            st.image(result["after_image"], use_container_width=True)
            st.caption("Lite Mode visual overlay only. This does not mark confirmed defect locations.")

        predicted_class = result["predicted_class"]
        card_class = {
            "good_weld": "good-card",
            "moderate_weld": "moderate-card",
            "bad_weld": "bad-card"
        }.get(predicted_class, "moderate-card")

        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        st.subheader("AI Classification Result")
        st.write(f"**Predicted Class:** {display_icon(predicted_class)} {display_label(predicted_class)}")
        st.write(f"**Confidence Score:** {result['confidence']:.2f}%")
        st.write(f"**Confidence Status:** {result['confidence_status']}")
        st.write(result["confidence_message"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Class Probability")

        prob_df = pd.DataFrame({
            "Class": [display_label(cls) for cls in result["probabilities"].keys()],
            "Probability (%)": [round(v, 2) for v in result["probabilities"].values()]
        })

        st.dataframe(prob_df, use_container_width=True)

        for cls, prob in result["probabilities"].items():
            st.write(f"{display_label(cls)}: {prob:.2f}%")
            st.progress(min(max(prob / 100, 0), 1))

        st.subheader("AI-Assisted Preliminary Rubric Score")

        rubric_df = pd.DataFrame({
            "Criteria": list(result["scores"].keys()),
            "Score": [f"{score} / 5" for score in result["scores"].values()]
        })

        st.dataframe(rubric_df, use_container_width=True)

        col_score1, col_score2 = st.columns(2)

        with col_score1:
            st.metric("Total Score", f"{result['total_score']} / {result['max_mark']}")

        with col_score2:
            st.metric("Percentage", f"{result['percentage']:.2f}%")

        st.subheader("Feedback")
        st.write(result["feedback"]["summary"])
        st.write(result["feedback"]["feedback"])
        st.info("Recommended Improvement: " + result["feedback"]["improvement"])

        if result["confidence"] < 60:
            st.error("Low confidence result. Instructor review is strongly required.")
        elif result["confidence"] < 80:
            st.warning("Moderate confidence result. Instructor review is recommended.")
        else:
            st.success("Prediction confidence is relatively high. Instructor verification is still required for final marking.")


# =========================
# PDF Report
# =========================

elif page == "PDF Report":
    st.header("PDF Report")

    result = st.session_state.get("assessment_result")

    if result is None:
        st.warning("No assessment result available. Please run analysis first.")
    else:
        st.write("Generate and download the AI-assisted visual welding assessment report.")

        st.write("**Predicted Class:**", display_label(result["predicted_class"]))
        st.write("**Confidence:**", f"{result['confidence']:.2f}%")
        st.write("**Score:**", f"{result['total_score']} / {result['max_mark']}")
        st.write("**Percentage:**", f"{result['percentage']:.2f}%")

        pdf = create_pdf_report(result)

        st.download_button(
            label="Download PDF Report",
            data=pdf,
            file_name="AI_Welgrade_Assessment_Report.pdf",
            mime="application/pdf"
        )


# =========================
# CQI Feedback
# =========================



elif page == "User Evaluation / CQI Feedback":
    st.header("User Evaluation / CQI Feedback")

    st.write(
        "Please complete the evaluation form below. "
        "Your feedback will be used for Continuous Quality Improvement (CQI) of AI Welgrade by PBS."
    )

    st.info(
        "If the embedded form does not load properly, please open the feedback form in a new tab."
    )

    st.link_button(
        "Open CQI Feedback Form in New Tab",
        "https://docs.google.com/forms/d/e/1FAIpQLSeCRriu80Pyo4lReS4_EvjOwczZJBIgp83hBwS7hMqpddbN5A/viewform?usp=header"
    )

    components.iframe(
        "https://docs.google.com/forms/d/e/1FAIpQLSeCRriu80Pyo4lReS4_EvjOwczZJBIgp83hBwS7hMqpddbN5A/viewform?embedded=true",
        height=1200,
        scrolling=True
    )


# =========================
# About
# =========================

elif page == "About & Limitation":
    st.header("About & Limitation")

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("About the Application")
    st.write(
        "AI Welgrade by PBS is a Lite Mode prototype for AI-assisted visual welding assessment. "
        "It analyses welding images and provides visual classification, preliminary rubric scoring and feedback."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="limit-card">', unsafe_allow_html=True)
    st.subheader("Important Limitation")
    st.write(
        "This system performs external visual image classification only. "
        "It does not assess internal penetration, internal fusion, metallurgical properties or actual weld strength."
    )
    st.write(
        "The generated rubric score is an AI-assisted preliminary score and must not replace instructor judgement, "
        "physical measurement, destructive testing or non-destructive testing."
    )
    st.write(
        "The After Analysis image uses a status overlay only. It does not indicate confirmed defect locations."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Model Information")
    st.write("Model type: MobileNetV2 transfer learning")
    st.write("Classification classes:", ", ".join([display_label(cls) for cls in class_names]))
    st.write("Mode: Lite Mode visual classification")


st.markdown(
    '<div class="footer-note">AI Welgrade by PBS | AI-assisted preliminary welding assessment for teaching and learning support</div>',
    unsafe_allow_html=True
)
