# 🚦 Intelligent Traffic Management System (ITMS)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](LICENSE)

An AI-powered, adaptive traffic signal control system that uses computer vision (**YOLOv8**) to dynamically optimize intersection green light timings based on real-time vehicle density and emergency prioritization.

---
![image alt](https://github.com/Jotheeswaranv/An-AI-Driven-Intelligent-Traffic-Signal-Management-System-Using-Deep-Reinforcement-Learning-/blob/c2440f005df795c10434ad09e678b5930c8492f7/cover_image.png)

## 🌟 Key Features

* **⚡ Real-Time Vehicle Detection:** Utilizes YOLOv8 to accurately track and count cars, bikes, trucks, and buses.
* **🚨 Emergency Vehicle Priority:** Instantly grants an immediate green light upon detecting critical vehicles like ambulances or fire trucks.
* **🧠 Density-Based Adaptive Timing:** Dynamic calculation of green light windows using a weighted density algorithm.
* **🧵 Multi-Lane Concurrency:** Parallel processing of 4 distinct traffic lanes utilizing `ThreadPoolExecutor` for zero-lag streaming.
* **📊 Live Web Dashboard:** Seamlessly stream annotated video frames, live vehicle counts, and simulated traffic signal states.
* **🔒 Secure Admin Controls:** SQLite-backed admin authentication using hashed password security.
* **📡 Instant Updates:** Uses Server-Sent Events (SSE) to stream data from the backend to the UI in real-time.

---
Presentation of Our Project: [Click here!](https://github.com/Jotheeswaranv/An-AI-Driven-Intelligent-Traffic-Signal-Management-System-Using-Deep-Reinforcement-Learning-/blob/769554c9f21c31939cf4fbc0d11a34231dfe3a7c/presentation%20for%20the%20project.pdf)

Our Conference: [Click Here!](https://github.com/Jotheeswaranv/An-AI-Driven-Intelligent-Traffic-Signal-Management-System-Using-Deep-Reinforcement-Learning-/blob/d93530d30948b2568b82156fe70e6693fa151bf1/An%20AI-Driven%20Intelligent%20Traffic%20Signal%20and%20Road%20Safety%20Management%20System%20Using%20Deep%20Reinforcement%20Learning.docx)

## 📁 Project Structure

```text
traffic_ai_system/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes/
│   │   ├── auth.py          # Login / logout authentication
│   │   ├── dashboard.py     # Home dashboard views
│   │   ├── upload.py        # 4-lane video ingestion
│   │   └── analysis.py      # Core analysis trigger & SSE stream
│   ├── models/
│   │   └── user.py          # SQLAlchemy User schema
│   ├── services/
│   │   ├── yolo_detector.py # Object detection & framing logic
│   │   ├── traffic_logic.py # Signal decision-making algorithms
│   │   └── video_processor.py # Multi-threaded lane management
│   ├── templates/           # Jinja2 Frontend layouts
│   └── static/              # Assets and processed data cache
├── instance/                # local SQLite DB path
├── .env                     # App configuration secrets
├── run.py                   # Production entry point
└── requirements.txt         # Project dependencies

🚀 Getting Started
1. Environment Setup
Clone the project repository and navigate to the root directory:

cd traffic_ai_system

Set up a clean isolated virtual environment:

# Create environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate

2. Install Dependencies

pip install -r requirements.txt

⚠️ Note: YOLOv8 (ultralytics) will automatically fetch the standard yolov8n.pt weights on its initial run. If hardware acceleration/YOLO is missing, the system gracefully degrades to a mock demonstration mode.

3. Configuration
Generate a .env file in the root directory and configure your administrative settings:

SECRET_KEY=itms-super-secret-key-2026
ADMIN_USERNAME=traffic-admin
ADMIN_PASSWORD=admin123

4. Run the Server

python run.py

Once initialized, visit the portal locally at http://127.0.0.1:5000 or via your network allocation address.

🔐 Default Access Credentials
Field           Value
Username      traffic-admin
Password      admin123

🛠️ Operational Workflow

1. Authentication: Log into the management console using your admin credentials.2. Initialize: Click Start Project from the master control screen.
3. Ingestion: Upload 4 synchronized traffic videos corresponding to 4 intersection lanes (.mp4, .avi, .mov).
4. Processing: Click Upload and Start to initiate the AI worker pipelines.
5. Monitor: Access the unified visual interface to track:
   Bounding box overlay displays.
   Live intersection state lights (Red 🔴 / Green 🟢).
   Calculated vehicle weights and traffic lane densities.
6. Audit: Review historical performance data inside the dynamic live log table at the bottom of the hub.

🧠 Core Algorithmic Logic

🟢 Normal Operation Mode
   Lane weights are assigned using a strict category hierarchy to determine 
   
   actual road pressure:

   $$\text{Density Score} = (\text{Cars} \times 1.0) + (\text{Bikes} \times 0.5) + (\text{Buses} \times 2.5) + (\text{Trucks} \times 2.0)$$
   
   1. The lane exhibiting the highest calculated density score is awarded the next green cycle phase.
   2. Green Light Duration Formula:
      $$\text{Duration} = \text{Base}(10s) + (\text{Density} \times 2)$$
      
   (Dynamically capped between a strict minimum of 5s and a maximum of 60s).

🚨 Emergency Priority Mode

   1. If an Ambulance or Fire Truck is successfully classified in any lane, the normal cycle overrides immediately.
   2. The requesting lane shifts instantly to Green for an extended clearing period ($60\text{ seconds}$).
   3. All conflicting intersection phases instantly force shift to Red until the emergency threshold drops back to zero.

📝 Technical Notes & Performance Tuning
    
    1. Frame Skipping: To optimize computing throughput, only every 5th frame is analyzed by the YOLO neural network model.
    2. Smoothing Metrics: Uses a continuous rolling average across the last 10 frames to prevent signal flickering from momentary visual occlusions.
    3. Observability: Full server operation reports, exceptions, and lifecycle actions are piped smoothly to itms.log.
