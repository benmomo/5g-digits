# License and Attribution

This repository contains materials for the Native AI Solutions Development (NAISD) course, including code, preprocessed datasets, and documentation.

Some files in this repository are **derived from an external dataset**.  
In accordance with the original license (CC BY 4.0), we acknowledge and credit the dataset authors below.

---

## 📦 UCA-EHAR Dataset Attribution

This project uses a **preprocessed subset** of the UCA-EHAR dataset.

**Original dataset information:**

- **Title:** UCA-EHAR: A Benchmark Dataset for Human Activity Recognition with Smart Glasses  
- **Authors:** As listed in the Zenodo entry  
- **Repository:** Zenodo  
- **Record:** https://zenodo.org/records/5659336  
- **DOI:** 10.5281/zenodo.5659336  
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

The original dataset contains raw sensor recordings collected from smart glasses equipped with an accelerometer, gyroscope, and barometer, across 20 participants performing daily activities.

---

## 🔧 Preprocessed Data Included in This Repository

For teaching purposes, this repository includes **derived NumPy files** created from the original dataset:

- `windows_X.npy` — pre-windowed IMU signal segments  
- `windows_y.npy` — corresponding activity labels  

These files are **not** the raw dataset.  
They are processed derivatives designed to simplify the learning activities in Unit 2.1.

The derivation steps include:

- slicing raw IMU signals into fixed-length windows  
- normalizing sensor channels  
- assigning numerical class labels  
- exporting as NumPy arrays

These derivative files may be redistributed under the terms of **CC BY 4.0**, provided that proper attribution to the original dataset authors is included.

---

## 📝 Required Attribution

If you reuse or redistribute the dataset derivatives, please include the following attribution:

> “This work uses a derivative of the UCA-EHAR dataset (Zenodo record 5659336), released under the Creative Commons Attribution 4.0 International License.  
> © Original authors as listed on the dataset's Zenodo page.”

---

## 📚 Repository License (Code + Documentation)

All code, notebooks, and original documentation in this repository (excluding dataset derivatives) are licensed under the **MIT License**.

A full copy of the MIT License is available in the repository’s `LICENSE` file.





