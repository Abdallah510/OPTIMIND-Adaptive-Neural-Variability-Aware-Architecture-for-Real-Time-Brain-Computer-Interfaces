OPTIMIND is an adaptive, low-latency, and memory-efficient brain–computer interface (BCI) architecture designed for real-time EEG-based emotion recognition under strict power and hardware constraints. The system exploits emotion-induced neural variability as a control signal rather than treating it as noise, enabling efficient online learning with minimal non-volatile memory (NVM) overhead.

The architecture integrates a fully pipelined EEG processing workflow, including signal preprocessing, FFT-based feature extraction, SVM-based inference, and confidence-aware adaptive learning. To support long-term deployment on implantable and edge BCI devices, OPTIMIND introduces an optimized memory hierarchy with feature caching, support-vector caching, predictive prefetching, and margin-based update filtering, significantly reducing unnecessary NVM writes while preserving accuracy.

OPTIMIND is evaluated using a Python-based system-level simulator on the EEGEmotions-27 dataset in a fully streaming configuration. Results demonstrate an average online classification accuracy of 81% with an end-to-end latency of 31.6 ms, while maintaining bounded model growth and low learning overhead. Compared to a baseline system without memory and learning optimizations, OPTIMIND achieves substantially lower latency and improved memory efficiency.

## Dataset Description

**EEGEmotions-27** is a dataset of raw EEG recordings collected during affective elicitation experiments. It contains EEG data from **88 participants**, where each participant experienced **27 distinct emotional states** while watching emotionally evocative video clips. The dataset is designed for emotion recognition research and supports both offline analysis and real-time, streaming-based brain–computer interface (BCI) experiments.

The raw EEG data is stored in plain text files under the `eeg_raw/` directory. Each file follows the naming convention:

For example, `1_5.0.txt` corresponds to **participant 1**, **emotion ID 5 (anger)**, and **trial 0**. Each file contains multichannel EEG time-series data recorded using an **Emotiv X headset** with **14 EEG channels**, sampled at **256 Hz**. Rows represent time samples, while columns correspond to EEG channels, making the data suitable for window-based feature extraction, classification, and online learning.

---

## Dataset Structure
- **File format:** `.txt`
- **Sampling rate:** 256 Hz
- **Channels:** 14 EEG channels (Emotiv X)
- **Labels:** Emotion ID encoded in the filename

---

## Emotion ID Mapping

| Emotion ID | Emotion Name |
|-----------:|--------------|
| 1  | Admiration |
| 2  | Adoration |
| 3  | Aesthetic Appreciation |
| 4  | Amusement |
| 5  | Anger |
| 6  | Anxiety |
| 7  | Awe |
| 8  | Awkwardness |
| 9  | Boredom |
| 10 | Calmness |
| 11 | Confusion |
| 12 | Craving |
| 13 | Disgust |
| 14 | Empathic Pain |
| 15 | Entrancement |
| 16 | Excitement |
| 17 | Fear |
| 18 | Horror |
| 19 | Interest |
| 20 | Joy |
| 21 | Nostalgia |
| 22 | Relief |
| 23 | Romance |
| 24 | Sadness |
| 25 | Satisfaction |
| 26 | Sexual Desire |
| 27 | Surprise |
