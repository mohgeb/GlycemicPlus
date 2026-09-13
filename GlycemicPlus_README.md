# GlycemicPlus

### Exploring whether wearable PPG signals can help identify hypoglycemic and hyperglycemic states without requiring continuous glucose monitoring at inference time.

> **Research prototype — not a medical device and not intended for treatment or insulin-dosing decisions.**

---

## Inspiration

Continuous glucose monitors such as Dexcom provide invaluable information for people managing diabetes, but they require specialized hardware, recurring consumables, and invasive sensing.

At the same time, millions of people already wear devices capable of collecting physiological signals such as photoplethysmography (PPG), heart rate, motion, and temperature.

GlycemicPlus started with a simple question:

> **Can physiological information already captured by wearable sensors contain enough signal to help identify glycemic abnormalities?**

Rather than attempting to directly replace a CGM, GlycemicPlus explores whether wearable PPG can serve as the foundation for a future non-invasive glycemic early-warning system.

---

## What It Does

GlycemicPlus analyzes wearable PPG signals and estimates the probability that a physiological window corresponds to:

- **Hypoglycemia**
- **In-range glucose**
- **Hyperglycemia**

The system uses CGM measurements only as **ground truth during training and evaluation**.

CGM glucose is **not provided to the prediction model as an input**.

The final prototype combines two complementary approaches:

### Gradient-Boosted Physiological Model

A CatBoost classifier analyzes engineered physiological characteristics derived from PPG, including:

- pulse timing
- estimated heart rate
- pulse interval variability
- signal morphology
- spectral power
- waveform statistics
- derivative characteristics

### Temporal Deep Learning Model

A Temporal Convolutional Network (TCN) processes the raw PPG waveform directly and learns temporal representations that handcrafted features may miss.

### Hybrid Inference

The CatBoost and TCN probability outputs are combined into a hybrid prediction:

```text
                     PPG Window
                         |
             +-----------+-----------+
             |                       |
             v                       v
     Engineered Features          Raw PPG
             |                       |
             v                       v
          CatBoost                  TCN
             |                       |
             +-----------+-----------+
                         |
                         v
                 Hybrid Prediction
                         |
              +----------+----------+
              |          |          |
              v          v          v
            Hypo      In Range    Hyper
