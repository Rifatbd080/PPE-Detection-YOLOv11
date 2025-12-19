# 🛡️ AI-Based PPE Detection for Restaurant Hygiene (Bangladesh)
**Developed by a student of Uttara University**

This project uses the **YOLOv11** architecture to automate hygiene monitoring in the Bangladeshi hospitality sector. It specifically detects **Masks, Hairnets, and Gloves** to ensure food safety standards are met in restaurant kitchens and food processing units.

## 🇧🇩 Context & Use Case
In Bangladesh, manual hygiene inspection is challenging. This system provides:
* **Real-time Monitoring:** Automates the supervision of kitchen staff.
* **Public Health:** Reduces foodborne illnesses by ensuring PPE compliance.
* **Smart Bangladesh:** Aligns with the national vision of integrating AI into daily industries.

## 🧠 Model Weights
The trained YOLOv11 model weights are stored on Google Drive due to file size limits:
👉 **[Download best.pt here](https://drive.google.com/file/d/1nY59S_or61Rj_0sBDAEjqNFxDFPvr-OJ/view?usp=sharing)**

## 🚀 Project Components
* `PPE_Detection_Training.ipynb`: The complete training workflow on Google Colab.
* `app.py`: Streamlit dashboard for real-time image and video inference.
* `requirements.txt`: Necessary libraries to run the environment.

## 🛠️ How to Run Locally
1. Clone the repo:
   ```bash
   git clone [https://github.com/Rifatbd080/PPE-Detection-YOLOv11.git](https://github.com/Rifatbd080/PPE-Detection-YOLOv11.git)
   cd PPE-Detection-YOLOv11

   pip install -r requirements.txt

   streamlit run app.py
