# ⚡ An Intelligent Energy Brain for Smart Buildings

**Team Project | Department of Computer Science, RKMVERI**  
**Instructor:** Br. Bhaswarachaitanya (Tamal Maharaj)  
**Team Members:**  
- 🧠 **Jahar Kumar Paul (B2430049)** — Machine & Deep Learning  
- 🤖 **Ayan Kumar Batabyal (B2530074)** — Reinforcement Learning  
- 🔬 **Sayan Goswami (B2530098)** — Reinforcement Learning, System Integration

---

## 📄 Project Report

[📥 **Download Project Proposal (PDF)**](https://github.com/sayan369369/Efficient-Energy-Brain-for-Smart-Buildings/blob/main/Infinity_ProjectProposal.pdf)

---
## 🌍 Overview

> “We do not inherit the Earth from our ancestors; we borrow it from our children.” — *American Proverb*

The **Efficient Energy Brain** is an intelligent control framework designed to optimize energy consumption in smart buildings while maintaining occupant comfort.  
By integrating **Machine Learning (ML)** and **Reinforcement Learning (RL)**, it dynamically predicts, adapts, and optimizes energy usage — particularly in **HVAC (Heating, Ventilation, and Air Conditioning)** systems.

Our vision is to make buildings truly intelligent — **sustainable, autonomous, and energy-efficient**.

---

## 💡 Motivation

Energy consumption in large buildings continues to increase rapidly.  
Among all utilities, **HVAC systems account for a major share** of total energy use.

This project addresses the need for:
- 🌱 Sustainable energy consumption  
- 🌡️ Thermal comfort management  
- 🤖 Autonomous, adaptive decision-making  

---

## 🧠 Proposed Framework

The system operates in **two key phases:**

### 🔹 Phase 1 — Deep Learning (Prediction)
A **Multi-Output Deep Neural Network (DNN)** is trained on building data to predict:
- Indoor and outdoor temperature  
- Occupancy levels  
- Energy demand and cost  

**Inputs:** Room size, number of occupants, outdoor/indoor temperature, energy price, desired comfort level  
**Outputs:** Short-term forecasts (e.g., 30-minute window) for optimal control decisions.

---

### 🔹 Phase 2 — Reinforcement Learning (Adaptive Control)
The RL agent continuously interacts with the environment to adjust HVAC parameters.

- Observes real-time data + predictions from the DL model  
- Adapts temperature and fan speed automatically  
- Applies **reward** for energy efficiency and **penalty** for discomfort or manual overrides  
- Learns continuously from new data every few minutes  

**Example:**  
If occupancy unexpectedly increases, the RL model increases cooling capacity while minimizing excess energy use.

---

## 🔄 Continuous Learning and Updates

Every few minutes, the system updates:
- Deep Learning weights (based on environmental feedback)  
- Reinforcement Learning policy (based on comfort-energy trade-off)  

This ensures the **Energy Brain** evolves dynamically, reflecting real-world conditions in near real time.

---

## ⚙️ Loophole & Solution

**Challenge:** Who sets the *initial AC temperature* before control begins?  
**Solution:**  
A lightweight, pre-trained **RL initialization model** determines the first setting automatically — ensuring consistent, efficient start-up even before adaptive control begins.

---

## 🎯 Outcomes

- 🔋 Reduction in total energy consumption  
- 🌡️ Maintained optimal indoor comfort  
- 🤖 Autonomous and scalable system design  
- ♻️ Step toward sustainable smart buildings  

This hybrid ML + RL approach functions as an **Energy Brain** — enabling intelligent energy decisions.

---



