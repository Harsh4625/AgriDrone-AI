"""
AgriDrone AI — Flask Backend
Run: python app.py
"""

import os
import io
import json
import base64
import logging
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path

import webbrowser
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from werkzeug.utils import secure_filename

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB
    UPLOAD_FOLDER=Path("static/uploads"),
    SECRET_KEY=os.urandom(24),
)
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}
IMG_SIZE    = 128
MODEL_PATH    = Path("model/saved_model/best_model.keras")
MODEL_PATH_FB = Path("model/saved_model/best_model_phase1.keras")
CLASS_MAP_PATH = Path("model/saved_model/class_map.json")

# ── Lazy Model Loading ────────────────────────────────────────
_model     = None
_class_map = None  # {0: "ClassName", ...}


def get_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists() and not MODEL_PATH_FB.exists():
        log.warning("Model file not found — running in DEMO mode.")
        return None
    try:
        import tensorflow as tf
        path = MODEL_PATH if MODEL_PATH.exists() else MODEL_PATH_FB
        _model = tf.keras.models.load_model(str(path))
        log.info(f"Model loaded from {path}")
    except Exception as exc:
        log.error(f"Failed to load model: {exc}")
        _model = None
    return _model


def get_class_map():
    global _class_map
    if _class_map is not None:
        return _class_map
    if CLASS_MAP_PATH.exists():
        with open(CLASS_MAP_PATH) as f:
            raw = json.load(f)
        # Keys may be strings ("0", "1", ...) — convert to int
        _class_map = {int(k): v for k, v in raw.items()}
    else:
        # Fallback default map (PlantVillage 38 classes)
        _class_map = {
            0:  "Apple___Apple_scab",
            1:  "Apple___Black_rot",
            2:  "Apple___Cedar_apple_rust",
            3:  "Apple___healthy",
            4:  "Blueberry___healthy",
            5:  "Cherry_(including_sour)___Powdery_mildew",
            6:  "Cherry_(including_sour)___healthy",
            7:  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            8:  "Corn_(maize)___Common_rust_",
            9:  "Corn_(maize)___Northern_Leaf_Blight",
            10: "Corn_(maize)___healthy",
            11: "Grape___Black_rot",
            12: "Grape___Esca_(Black_Measles)",
            13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            14: "Grape___healthy",
            15: "Orange___Haunglongbing_(Citrus_greening)",
            16: "Peach___Bacterial_spot",
            17: "Peach___healthy",
            18: "Pepper,_bell___Bacterial_spot",
            19: "Pepper,_bell___healthy",
            20: "Potato___Early_blight",
            21: "Potato___Late_blight",
            22: "Potato___healthy",
            23: "Raspberry___healthy",
            24: "Soybean___healthy",
            25: "Squash___Powdery_mildew",
            26: "Strawberry___Leaf_scorch",
            27: "Strawberry___healthy",
            28: "Tomato___Bacterial_spot",
            29: "Tomato___Early_blight",
            30: "Tomato___Late_blight",
            31: "Tomato___Leaf_Mold",
            32: "Tomato___Septoria_leaf_spot",
            33: "Tomato___Spider_mites Two-spotted_spider_mite",
            34: "Tomato___Target_Spot",
            35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            36: "Tomato___Tomato_mosaic_virus",
            37: "Tomato___healthy",
        }
    return _class_map


# ── Treatment Database — 38 PlantVillage Classes ─────────────
TREATMENTS = {
    # ── APPLE ─────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis, producing dark, scabby lesions on leaves and fruit.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Apply sulfur-based fungicides at bud break", "Rake and destroy fallen leaves", "Neem oil spray @ 3ml/L every 10 days"],
        "inorganic": ["Captan 50% WP @ 2.5g/L", "Mancozeb 75% WP @ 2.5g/L", "Myclobutanil 40% WP @ 1g/L"],
        "prevention":["Plant resistant varieties", "Prune for air circulation", "Avoid overhead irrigation"],
        "pest_risk": "Low", "nutrient_note": "Maintain balanced calcium — improves fruit skin integrity.",
    },
    "Apple___Black_rot": {
        "description": "Black rot is caused by Botryosphaeria obtusa, causing frogeye leaf spots and mummified fruit.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Remove mummified fruit and cankers", "Copper hydroxide spray @ 3g/L", "Bordeaux mixture application"],
        "inorganic": ["Thiophanate-methyl 70% WP @ 1.5g/L", "Captan 80% WG @ 2g/L"],
        "prevention":["Prune dead/cankered wood", "Sanitize pruning tools", "Avoid wounding bark"],
        "pest_risk": "Low", "nutrient_note": "Excess nitrogen promotes susceptibility — balance with potassium.",
    },
    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust is caused by Gymnosporangium juniperi-virginianae, producing orange spots on leaves.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Remove nearby juniper/cedar hosts", "Sulfur spray @ 3g/L at pink bud stage", "Neem oil preventive spray"],
        "inorganic": ["Myclobutanil 40% WP @ 1g/L", "Propiconazole 25% EC @ 1ml/L"],
        "prevention":["Plant resistant apple varieties", "Avoid planting near cedar trees"],
        "pest_risk": "Low", "nutrient_note": "Adequate potassium strengthens cell walls against rust penetration.",
    },
    "Apple___healthy": {
        "description": "The apple crop appears healthy with no visible disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Continue preventive sulfur sprays", "Apply compost mulch around base"],
        "inorganic": ["Balanced NPK foliar spray fortnightly"],
        "prevention":["Regular scouting", "Maintain orchard sanitation"],
        "pest_risk": "Low", "nutrient_note": "Ensure adequate boron for fruit set.",
    },
    # ── BLUEBERRY ──────────────────────────────────────────────
    "Blueberry___healthy": {
        "description": "The blueberry crop is healthy with no disease signs.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Apply pine bark mulch to maintain acidic pH", "Preventive copper spray"],
        "inorganic": ["Acidic fertilizer (ammonium sulfate)"],
        "prevention":["Maintain soil pH 4.5–5.5", "Avoid alkaline water"],
        "pest_risk": "Low", "nutrient_note": "Blueberries need acidic soil — test pH regularly.",
    },
    # ── CHERRY ─────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Powdery mildew on cherry causes white powdery coating on young leaves and shoots.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Potassium bicarbonate @ 5g/L spray", "Neem oil @ 3ml/L weekly", "Baking soda solution @ 5g/L"],
        "inorganic": ["Myclobutanil 40% WP @ 1g/L", "Trifloxystrobin 50% WG @ 0.5g/L"],
        "prevention":["Prune for canopy airflow", "Avoid excess nitrogen fertilization"],
        "pest_risk": "Low", "nutrient_note": "Reduce nitrogen — excess promotes succulent tissue susceptible to mildew.",
    },
    "Cherry_(including_sour)___healthy": {
        "description": "The cherry crop is healthy with no disease symptoms detected.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Preventive copper sprays post-harvest", "Compost application"],
        "inorganic": ["Balanced NPK fertilization"],
        "prevention":["Monitor for cherry fruit fly", "Maintain orchard hygiene"],
        "pest_risk": "Low", "nutrient_note": "Calcium sprays reduce fruit cracking.",
    },
    # ── CORN ───────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray leaf spot is a fungal disease causing rectangular tan lesions parallel to leaf veins.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Crop rotation with non-host crops", "Trichoderma-based biocontrol spray", "Destroy crop residues post-harvest"],
        "inorganic": ["Azoxystrobin 23% SC @ 1ml/L", "Propiconazole 25% EC @ 1ml/L", "Pyraclostrobin 20% WG @ 1g/L"],
        "prevention":["Plant resistant hybrids", "Avoid dense planting", "Minimum tillage"],
        "pest_risk": "Low", "nutrient_note": "Adequate silicon foliar spray improves resistance.",
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Common rust causes brick-red pustules on both leaf surfaces, reducing photosynthesis.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Neem oil spray @ 5ml/L", "Sulfur dust application", "Early planting to avoid peak rust season"],
        "inorganic": ["Mancozeb 75% WP @ 2.5g/L", "Tebuconazole 25.9% EC @ 1ml/L"],
        "prevention":["Use rust-resistant corn hybrids", "Scout fields weekly during humid conditions"],
        "pest_risk": "Low", "nutrient_note": "High nitrogen can worsen rust — maintain N:K balance.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Northern leaf blight causes large cigar-shaped tan lesions on corn leaves, caused by Exserohilum turcicum.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Bacillus subtilis-based spray", "Crop rotation", "Destroy infected debris"],
        "inorganic": ["Azoxystrobin + Propiconazole (Quilt Xcel) @ 1ml/L", "Mancozeb 75% WP @ 2.5g/L"],
        "prevention":["Plant tolerant hybrids", "Avoid late planting"],
        "pest_risk": "Low", "nutrient_note": "Silica supplementation strengthens leaf tissue.",
    },
    "Corn_(maize)___healthy": {
        "description": "The corn crop is healthy with no visible disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Compost side-dressing at knee-high stage"],
        "inorganic": ["Urea top-dressing at V6 stage"],
        "prevention":["Scout for fall armyworm and aphids"],
        "pest_risk": "Low", "nutrient_note": "Corn is a heavy feeder — ensure adequate N, P, K, and zinc.",
    },
    # ── GRAPE ──────────────────────────────────────────────────
    "Grape___Black_rot": {
        "description": "Grape black rot (Guignardia bidwellii) causes brown leaf lesions and shriveled, mummified berries.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Remove mummified berries and infected canes", "Bordeaux mixture @ 1% spray", "Copper oxychloride @ 3g/L"],
        "inorganic": ["Myclobutanil 40% WP @ 1g/L", "Mancozeb 75% WP @ 2.5g/L", "Tebuconazole 25.9% EC @ 1ml/L"],
        "prevention":["Remove all mummies before bud break", "Prune for canopy airflow", "Avoid leaf wetness"],
        "pest_risk": "Low", "nutrient_note": "Potassium improves berry skin integrity and disease resistance.",
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Esca (Black Measles) is a complex wood disease causing interveinal chlorosis, necrosis, and sudden vine collapse.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Remove and destroy infected wood", "Paint pruning wounds with Trichoderma paste", "Avoid large pruning wounds"],
        "inorganic": ["Sodium arsenite (where legally permitted) wound paint", "Fosetyl-aluminium foliar spray"],
        "prevention":["Use double pruning technique", "Prune during dry weather", "Protect wounds immediately"],
        "pest_risk": "Low", "nutrient_note": "Avoid water stress — maintain consistent drip irrigation.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Isariopsis leaf spot causes dark brown spots with yellow halos on grape leaves, leading to defoliation.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Copper-based fungicide @ 3g/L", "Neem oil @ 3ml/L every 10 days", "Remove infected leaves"],
        "inorganic": ["Mancozeb 75% WP @ 2.5g/L", "Captan 50% WP @ 2.5g/L"],
        "prevention":["Improve canopy management", "Avoid overhead irrigation"],
        "pest_risk": "Low", "nutrient_note": "Magnesium deficiency worsens blight — apply Epsom salt foliar spray.",
    },
    "Grape___healthy": {
        "description": "The grape vine appears healthy with no visible disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Preventive Bordeaux mixture pre-season", "Compost mulching"],
        "inorganic": ["Balanced fertilizer at bud break"],
        "prevention":["Scout for mealybugs and leafhoppers", "Maintain trellis system"],
        "pest_risk": "Low", "nutrient_note": "Ensure adequate potassium for berry quality.",
    },
    # ── ORANGE ─────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Citrus greening (HLB) is a devastating bacterial disease spread by the Asian citrus psyllid, causing blotchy mottling and lopsided bitter fruit.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Remove and destroy infected trees immediately", "Control psyllid vector with neem oil @ 5ml/L", "Release Tamarixia radiata (parasitic wasp) for biocontrol"],
        "inorganic": ["Imidacloprid systemic insecticide for psyllid", "Oxytetracycline trunk injection (where permitted)", "Zinc + Manganese foliar spray to manage symptoms"],
        "prevention":["Use certified disease-free planting material", "Install psyllid exclusion nets", "Quarantine new plant material"],
        "pest_risk": "High", "nutrient_note": "HLB disrupts nutrient uptake — apply chelated micronutrient sprays.",
    },
    # ── PEACH ──────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "description": "Bacterial spot (Xanthomonas arboricola) causes water-soaked spots on leaves, fruit, and twigs.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Copper hydroxide spray @ 3g/L before rain", "Remove infected twigs", "Kaolin clay spray to repel insects"],
        "inorganic": ["Oxytetracycline 17% SP @ 0.3g/L", "Copper oxychloride + Mancozeb tank mix"],
        "prevention":["Plant resistant varieties", "Avoid overhead irrigation", "Prune for air circulation"],
        "pest_risk": "Low", "nutrient_note": "Calcium sprays improve fruit skin resistance to bacterial entry.",
    },
    "Peach___healthy": {
        "description": "The peach tree appears healthy with no visible disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Preventive copper sprays post-harvest", "Compost mulch around base"],
        "inorganic": ["Balanced NPK at pink bud stage"],
        "prevention":["Scout for peach twig borer and oriental fruit moth"],
        "pest_risk": "Low", "nutrient_note": "Peach needs zinc and boron for proper fruit development.",
    },
    # ── PEPPER BELL ────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial spot (Xanthomonas vesicatoria) causes water-soaked spots on pepper leaves and fruit, leading to defoliation.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Copper oxychloride spray @ 3g/L", "Remove infected plant material", "Bacillus amyloliquefaciens biocontrol"],
        "inorganic": ["Streptocycline 90% SP @ 0.1g/L + Copper oxychloride", "Mancozeb 75% WP @ 2.5g/L"],
        "prevention":["Use certified disease-free seeds", "Avoid working in wet fields", "Crop rotation with non-solanaceous crops"],
        "pest_risk": "Low", "nutrient_note": "Calcium prevents blossom end rot — apply calcium nitrate @ 2g/L.",
    },
    "Pepper,_bell___healthy": {
        "description": "The pepper crop is healthy with no visible disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Neem cake soil application", "Preventive copper spray monthly"],
        "inorganic": ["High-potassium fertilizer at fruiting stage"],
        "prevention":["Scout for thrips and aphids weekly", "Use yellow sticky traps"],
        "pest_risk": "Low", "nutrient_note": "Peppers need high potassium and calcium at fruiting — monitor closely.",
    },
    # ── POTATO ─────────────────────────────────────────────────
    "Potato___Early_blight": {
        "description": "Early blight (Alternaria solani) causes dark brown target-spot lesions on lower potato leaves.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Neem oil spray @ 3ml/L weekly", "Copper oxychloride @ 3g/L", "Remove lower infected leaves"],
        "inorganic": ["Mancozeb 75% WP @ 2.5g/L", "Chlorothalonil 75% WP @ 2g/L", "Azoxystrobin 23% SC @ 1ml/L"],
        "prevention":["Use certified seed potatoes", "Avoid overhead irrigation", "Proper plant spacing"],
        "pest_risk": "Low", "nutrient_note": "Potassium deficiency worsens early blight — ensure adequate K.",
    },
    "Potato___Late_blight": {
        "description": "Late blight (Phytophthora infestans) is a devastating oomycete causing rapid brown rot of leaves and tubers.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Copper hydroxide spray @ 3g/L at first sign", "Destroy infected haulms immediately", "Avoid planting near infected fields"],
        "inorganic": ["Metalaxyl + Mancozeb @ 2.5g/L", "Cymoxanil 8% + Mancozeb 64% WP @ 2g/L", "Dimethomorph 50% WP @ 1g/L"],
        "prevention":["Use late-blight resistant varieties (Kufri Jyoti)", "Ridge planting for better drainage", "Destroy cull piles"],
        "pest_risk": "Low", "nutrient_note": "Phosphorus promotes tuber health — ensure adequate P levels.",
    },
    "Potato___healthy": {
        "description": "The potato crop is healthy with no visible disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Trichoderma soil drench at planting", "Compost application"],
        "inorganic": ["NPK 10:26:26 at earthing-up stage"],
        "prevention":["Scout for aphids (late blight vectors)", "Monitor soil moisture"],
        "pest_risk": "Low", "nutrient_note": "Potatoes are heavy potassium feeders — ensure K levels are high.",
    },
    # ── RASPBERRY ──────────────────────────────────────────────
    "Raspberry___healthy": {
        "description": "The raspberry crop is healthy with no visible disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Preventive sulfur spray at bud break", "Compost mulching"],
        "inorganic": ["Ammonium nitrate top-dressing post-harvest"],
        "prevention":["Remove old floricanes after harvest", "Scout for spider mites in hot weather"],
        "pest_risk": "Low", "nutrient_note": "Raspberries need magnesium — apply dolomitic lime if pH is low.",
    },
    # ── SOYBEAN ────────────────────────────────────────────────
    "Soybean___healthy": {
        "description": "The soybean crop is healthy with no visible disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Rhizobium inoculant at planting", "Neem cake soil application"],
        "inorganic": ["Phosphorus and potassium basal dose"],
        "prevention":["Scout for soybean aphids and stink bugs", "Monitor for soybean rust"],
        "pest_risk": "Low", "nutrient_note": "Soybeans fix nitrogen — avoid excess N application.",
    },
    # ── SQUASH ─────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "description": "Powdery mildew on squash causes white powdery spots on leaves, reducing photosynthesis and yield.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Potassium bicarbonate @ 5g/L spray", "Neem oil @ 3ml/L weekly", "Milk spray (1:9 ratio with water)"],
        "inorganic": ["Myclobutanil 40% WP @ 1g/L", "Trifloxystrobin 50% WG @ 0.5g/L"],
        "prevention":["Plant resistant varieties", "Improve air circulation", "Avoid evening irrigation"],
        "pest_risk": "Low", "nutrient_note": "Excess nitrogen promotes succulent growth — lower N at fruiting.",
    },
    # ── STRAWBERRY ─────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "description": "Strawberry leaf scorch (Diplocarpon earliana) causes small purplish spots that enlarge and turn brown, scorching leaf margins.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Remove infected leaves promptly", "Copper oxychloride spray @ 3g/L", "Neem oil preventive spray"],
        "inorganic": ["Captan 50% WP @ 2.5g/L", "Myclobutanil 40% WP @ 1g/L"],
        "prevention":["Plant certified disease-free runners", "Avoid overhead irrigation", "Renovate beds annually"],
        "pest_risk": "Low", "nutrient_note": "Potassium improves disease resistance in strawberries.",
    },
    "Strawberry___healthy": {
        "description": "The strawberry crop is healthy with no visible disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Straw mulch to prevent soil-splash diseases", "Compost tea spray"],
        "inorganic": ["High-potassium fertilizer at fruiting"],
        "prevention":["Scout for two-spotted spider mites in hot weather"],
        "pest_risk": "Low", "nutrient_note": "Boron is critical for strawberry fruit set and quality.",
    },
    # ── TOMATO ─────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot (Xanthomonas vesicatoria) causes water-soaked lesions on tomato leaves, stems, and fruit.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Copper oxychloride spray @ 3g/L", "Bacillus subtilis biocontrol spray", "Remove infected plant material"],
        "inorganic": ["Streptocycline 90% SP @ 0.1g/L", "Mancozeb 75% WP @ 2.5g/L"],
        "prevention":["Use resistant varieties", "Avoid working in wet conditions", "Drip irrigation"],
        "pest_risk": "Low", "nutrient_note": "Calcium prevents blossom end rot — apply calcium nitrate @ 2g/L.",
    },
    "Tomato___Early_blight": {
        "description": "Early blight (Alternaria solani) causes dark concentric ring lesions on lower tomato leaves, progressing upward.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Neem oil spray @ 3ml/L", "Copper oxychloride @ 3g/L", "Remove lower infected leaves"],
        "inorganic": ["Mancozeb 75% WP @ 2.5g/L", "Chlorothalonil 75% WP @ 2g/L", "Azoxystrobin 23% SC @ 1ml/L"],
        "prevention":["Stake plants for air circulation", "Mulch to prevent soil splash", "Crop rotation"],
        "pest_risk": "Low", "nutrient_note": "Adequate potassium and phosphorus improve plant resistance.",
    },
    "Tomato___Late_blight": {
        "description": "Late blight (Phytophthora infestans) rapidly destroys tomato foliage and fruit in cool, wet conditions.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Copper hydroxide @ 3g/L at first sign", "Destroy infected plants immediately", "Avoid overhead irrigation"],
        "inorganic": ["Metalaxyl + Mancozeb @ 2.5g/L", "Cymoxanil + Mancozeb @ 2g/L", "Dimethomorph 50% WP @ 1g/L"],
        "prevention":["Plant resistant varieties", "Scout after cool wet weather", "Remove volunteer tomato plants"],
        "pest_risk": "Low", "nutrient_note": "Phosphorus strengthens plant immune response — ensure adequate P.",
    },
    "Tomato___Leaf_Mold": {
        "description": "Tomato leaf mold (Passalora fulva) causes pale greenish-yellow spots on upper leaf surface with olive-brown mold below.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Improve greenhouse ventilation", "Neem oil spray @ 3ml/L", "Remove badly infected leaves"],
        "inorganic": ["Chlorothalonil 75% WP @ 2g/L", "Mancozeb 75% WP @ 2.5g/L", "Iprodione 50% WP @ 2g/L"],
        "prevention":["Reduce humidity below 85%", "Space plants for airflow", "Avoid leaf wetness"],
        "pest_risk": "Low", "nutrient_note": "Potassium improves stomatal control, reducing leaf surface wetness.",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes numerous small, circular spots with dark borders and light centers on tomato leaves.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Remove infected lower leaves", "Copper oxychloride spray @ 3g/L", "Neem oil @ 3ml/L weekly"],
        "inorganic": ["Mancozeb 75% WP @ 2.5g/L", "Chlorothalonil 75% WP @ 2g/L"],
        "prevention":["Mulch around plants", "Stake plants", "Rotate with non-solanaceous crops"],
        "pest_risk": "Low", "nutrient_note": "Avoid over-watering — well-drained soil reduces disease severity.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Two-spotted spider mites cause stippling and bronzing of tomato leaves. Fine webbing visible under severe infestation.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Neem oil + soap spray @ 5ml/L + 2ml/L", "Release Phytoseiulus persimilis (predatory mite)", "Strong water jets on leaf undersides"],
        "inorganic": ["Abamectin 1.8% EC @ 0.5ml/L", "Spiromesifen 22.9% SC @ 0.9ml/L", "Fenpyroximate 5% EC @ 1ml/L"],
        "prevention":["Maintain adequate moisture", "Avoid dusty conditions", "Monitor in hot dry weather"],
        "pest_risk": "High", "nutrient_note": "Water-stressed plants are more susceptible — maintain consistent irrigation.",
    },
    "Tomato___Target_Spot": {
        "description": "Target spot (Corynespora cassiicola) causes brown circular lesions with concentric rings on tomato foliage and fruit.",
        "severity": "Medium", "severity_color": "#FF8C00",
        "organic":   ["Copper oxychloride spray @ 3g/L", "Remove infected leaves", "Improve plant spacing"],
        "inorganic": ["Azoxystrobin 23% SC @ 1ml/L", "Boscalid + Pyraclostrobin @ 1g/L"],
        "prevention":["Avoid high humidity conditions", "Crop rotation", "Use drip irrigation"],
        "pest_risk": "Low", "nutrient_note": "Balanced nutrition reduces stress and disease susceptibility.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "TYLCV is a viral disease spread by whiteflies, causing yellowing, upward curling of leaves, and severe yield loss.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Control whitefly vectors with neem oil @ 5ml/L", "Yellow sticky traps @ 10/acre", "Remove infected plants immediately"],
        "inorganic": ["Imidacloprid 17.8% SL @ 0.5ml/L for whitefly", "Thiamethoxam 25% WG @ 0.3g/L"],
        "prevention":["Use TYLCV-resistant varieties", "Silver reflective mulch to deter whitefly", "Install insect-proof nets"],
        "pest_risk": "High", "nutrient_note": "Infected plants cannot be cured — focus on prevention.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Tomato mosaic virus (ToMV) causes mosaic mottling, distortion, and stunting. Spread by contact and aphids.",
        "severity": "High", "severity_color": "#FF4444",
        "organic":   ["Remove and destroy infected plants", "Wash hands with soap before handling", "Control aphid vectors with neem oil"],
        "inorganic": ["No cure — focus on vector control with Imidacloprid"],
        "prevention":["Use ToMV-resistant varieties", "Sanitize tools with 10% bleach", "Control aphids and thrips"],
        "pest_risk": "Medium", "nutrient_note": "Keep plants vigorous with balanced nutrition to slow virus progression.",
    },
    "Tomato___healthy": {
        "description": "The tomato crop is healthy with vibrant foliage and no disease symptoms.",
        "severity": "None", "severity_color": "#00CC66",
        "organic":   ["Preventive neem oil spray fortnightly", "Compost side-dressing"],
        "inorganic": ["High-K fertilizer at flowering/fruiting stage"],
        "prevention":["Scout weekly for early blight, aphids, whitefly"],
        "pest_risk": "Low", "nutrient_note": "Tomatoes need calcium and magnesium — apply dolomite lime if deficient.",
    },
}

# Default for unknown classes
DEFAULT_TREATMENT = {
    "description": "An anomaly was detected in the crop. Please consult a local agricultural expert for further assessment.",
    "severity": "Unknown",
    "severity_color": "#888888",
    "organic":   ["Consult local Krishi Vigyan Kendra (KVK)", "Collect samples for lab diagnosis"],
    "inorganic": ["Wait for professional diagnosis before applying chemicals"],
    "prevention":["Isolate the affected area", "Document the symptoms with photos"],
    "pest_risk": "Unknown",
    "nutrient_note": "Run a soil test to check nutrient deficiencies.",
}


# ── Helpers ───────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def demo_prediction(class_map: dict) -> dict:
    """Simulated prediction when model is absent (for UI testing)."""
    import random
    classes = list(class_map.values())
    probs   = np.random.dirichlet(np.ones(len(classes)) * 0.5)
    top_idx = int(np.argmax(probs))
    top_cls = classes[top_idx]
    return {
        "demo_mode": True,
        "predicted_class": top_cls,
        "display_name": top_cls.replace("_", " "),
        "confidence": float(probs[top_idx]),
        "class_probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
        "treatment": TREATMENTS.get(top_cls, DEFAULT_TREATMENT),
        "timestamp": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
    }


def run_inference(img_array: np.ndarray) -> dict:
    model     = get_model()
    class_map = get_class_map()

    if model is None:
        return demo_prediction(class_map)

    preds   = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_cls = class_map.get(top_idx, "Unknown")

    class_probs = {
        class_map.get(i, f"Class_{i}"): float(preds[i])
        for i in range(len(preds))
    }

    return {
        "demo_mode": False,
        "predicted_class": top_cls,
        "display_name": top_cls.replace("_", " "),
        "confidence": float(preds[top_idx]),
        "class_probabilities": class_probs,
        "treatment": TREATMENTS.get(top_cls, DEFAULT_TREATMENT),
        "timestamp": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
    }


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate ─────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file received."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}"}), 400

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"success": False, "error": "Empty file uploaded."}), 400

        # Validate it's a real image
        Image.open(io.BytesIO(image_bytes)).verify()

        img_array = preprocess_image(image_bytes)
        result    = run_inference(img_array)
        result["success"] = True
        return jsonify(result), 200

    except Exception:
        log.error(traceback.format_exc())
        return jsonify({"success": False, "error": "Failed to process the image. Please try a valid crop image."}), 500


@app.route("/report", methods=["POST"])
def generate_report():
    """Generate a PDF report from prediction results."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        GREEN  = colors.HexColor("#00CC66")
        DARK   = colors.HexColor("#0D1F0F")

        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     textColor=DARK, fontSize=20, spaceAfter=6)
        h2_style    = ParagraphStyle("h2",    parent=styles["Heading2"],
                                     textColor=GREEN, fontSize=13, spaceBefore=12, spaceAfter=4)
        body_style  = ParagraphStyle("body",  parent=styles["Normal"], fontSize=10, leading=14)

        story = []
        story.append(Paragraph("🌿 AgriDrone AI — Crop Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {data.get('timestamp', datetime.now().strftime('%d %b %Y, %H:%M:%S'))}", body_style))
        story.append(HRFlowable(width="100%", color=GREEN, spaceAfter=12))

        story.append(Paragraph("Detection Result", h2_style))
        tbl_data = [
            ["Detected Condition", data.get("display_name", "—")],
            ["Confidence Score", f"{data.get('confidence', 0)*100:.2f}%"],
            ["Severity Level",   data.get("treatment", {}).get("severity", "—")],
            ["Pest Risk",        data.get("treatment", {}).get("pest_risk", "—")],
        ]
        tbl = Table(tbl_data, colWidths=[6*cm, 10*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8F5E9")),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
            ("PADDING",    (0,0), (-1,-1), 6),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

        treatment = data.get("treatment", {})
        story.append(Paragraph("Description", h2_style))
        story.append(Paragraph(treatment.get("description", "—"), body_style))

        for section, key in [("Organic Treatments", "organic"),
                              ("Chemical / Inorganic Treatments", "inorganic"),
                              ("Prevention Strategies", "prevention")]:
            items = treatment.get(key, [])
            if items:
                story.append(Paragraph(section, h2_style))
                for item in items:
                    story.append(Paragraph(f"• {item}", body_style))

        story.append(Spacer(1, 8))
        story.append(Paragraph("Nutrient Advisory", h2_style))
        story.append(Paragraph(treatment.get("nutrient_note", "—"), body_style))

        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", color=colors.grey))
        story.append(Paragraph("Report generated by AgriDrone AI System — Dr. D. Y. Patil Institute of Technology, Pimpri, Pune", 
                               ParagraphStyle("footer", parent=styles["Normal"], fontSize=8,
                                              textColor=colors.grey, alignment=TA_CENTER)))

        doc.build(story)
        buf.seek(0)

        fname = f"AgriDrone_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(buf, mimetype="application/pdf",
                         as_attachment=True, download_name=fname)

    except ImportError:
        return jsonify({"success": False, "error": "reportlab not installed. Run: pip install reportlab"}), 500
    except Exception:
        log.error(traceback.format_exc())
        return jsonify({"success": False, "error": "Failed to generate report."}), 500


@app.route("/health")
def health():
    model_loaded = get_model() is not None
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "mode": "inference" if model_loaded else "demo",
        "timestamp": datetime.now().isoformat(),
    })


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting AgriDrone AI Server...")
    log.info(f"Model path : {MODEL_PATH}")
    log.info(f"Model found: {MODEL_PATH.exists() or MODEL_PATH_FB.exists()}")
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
