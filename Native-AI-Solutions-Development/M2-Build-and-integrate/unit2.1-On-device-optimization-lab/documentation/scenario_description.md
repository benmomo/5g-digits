# Smart-Glasses Human Activity Recognition (HAR) Scenario

This project uses a realistic scenario where workers in an indoor industrial environment wear **smart safety glasses** equipped with small motion sensors. These glasses help improve workplace safety, ergonomics, and situational awareness.

The goal is to build and optimize a **machine learning model** that can recognize the worker’s activity directly on the device (on-device AI).

---

## 🔧 Sensors in the Smart Glasses

The glasses contain:

- **3-axis accelerometer** – measures movement along x/y/z  
- **3-axis gyroscope** – measures rotation/orientation  
- **Barometer** – measures air pressure (optional depending on device)

These sensors record continuous motion signals while the worker performs daily activities.

---

## 🧑‍🏭 Industrial Context

Workers operate inside large indoor industrial facilities (assembly, warehouse, logistics, maintenance). Examples of activities include:

- Walking between workstations  
- Carrying materials  
- Sitting or standing during tasks  
- Climbing up or down stairs  
- Taking short breaks (e.g., drinking water)  

These activities generate distinct motion patterns that can be detected by an AI model.

---

## 🎯 Activities to Recognize

The model must classify **eight activities**, which come from the UCA-EHAR dataset:

1. **Standing**  
2. **Sitting**  
3. **Walking**  
4. **Walking Upstairs**  
5. **Walking Downstairs**  
6. **Running**  
7. **Lying** (e.g., fall or accident)  
8. **Drinking**

Each activity has a characteristic motion signature in the sensor data.

---

## ⚠️ Why Activity Recognition Matters

The smart glasses can support important safety and ergonomics features:

### Safety Use Cases
- Detect **running** in restricted areas  
- Detect possible **falls** (transition to LYING)  
- Detect **sudden abnormal motion**

### Ergonomics Use Cases
- Identify long **static postures** (standing or sitting)  
- Track patterns for **fatigue management**  
- Provide optional **micro-break reminders**

### Real-Time Feedback
Because safety events can unfold quickly, the classification must run in **real time** on a small device.

This leads to technical constraints such as:

- Inference latency under **20 ms**  
- Model size under **1 MB**  
- CPU-only execution (no GPU)  
- Low memory usage  
- No internet or cloud dependency

These constraints shape the optimization tasks in this unit.

---

## 🧠 Your Role in This Project

As a developer working on Native AI Solutions, you will:

- Load a **baseline HAR model**  
- Measure its performance (latency, size, accuracy)  
- Apply optimization techniques (quantization, pruning)  
- Produce a final **optimized on-device model**  
- Prepare the optimized model for integration in the next unit

This hands-on work mirrors real-world edge AI development.

---

## 📦 Dataset Used in This Project

This project uses a **preprocessed subset** of the UCA-EHAR dataset:

- **Title:** *UCA-EHAR: A Benchmark Dataset for Human Activity Recognition with Smart Glasses*  
- **Zenodo Record:** https://zenodo.org/records/5659336  
- **License:** **CC BY 4.0** (reuse allowed with attribution)

The original dataset contains raw sensor recordings for the 8 activities. For this course, we use **pre-windowed NumPy arrays** (`windows_X.npy`, `windows_y.npy`) that simplify the training and optimization tasks.

---

## 🏗️ What You Will Build in This Unit

During Unit 2.1 you will:

- Inspect the sensor data  
- Load and analyze the baseline HAR model  
- Measure how fast and how large it is  
- Apply **quantization** (float32 → int8)  
- Apply **pruning** (removing redundant weights)  
- Evaluate improvements  
- Export a **final optimized model** ready for deployment

Your final output will be a one-page **Optimization Report** summarizing improvements in speed, memory, and model size.


