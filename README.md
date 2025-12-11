# Efficient Model Compression for Conversion Prediction  
**COMS E6998 – High-Performance Machine Learning (Final Project)**  
Columbia University  

## 📌 Overview

This repository contains our implementation and experimentation pipeline for compressing large-scale **conversion prediction models** used in Google Ads. These models face strict constraints on **latency**, **memory footprint**, and **serving cost**, motivating the need for efficient student architectures that can preserve predictive performance while reducing inference overhead.

Our work evaluates the effectiveness of several model-compression strategies:

- **Knowledge Distillation (KD)**
- **Feature-Based Distillation (intermediate-layer matching)**
- **INT8 Post-Training Quantization (PTQ)**
- **Prototype pruning approaches**

The combined KD + INT8 pipeline produced the strongest student model, demonstrating high efficiency without degrading predictive reliability.

## 🚀 Key Results

- **–0.15% change in Poisson Log Loss** (near-teacher performance)
- **0.14 ms improvement in critical-path serving latency**
- **Reduced TPU compute usage**
- **Positive advertiser ROI** in preliminary live-traffic tests

These results are consistent with the findings summarized in our final report.

## 📂 Repository Structure

```
COMS-E6998-final-project/
│
├── knowledge_distillation/
│ ├── distillation_v1.py
│ └── distillation_v2.py
│
├── pruning/
│ ├── pruning_v1.py
│ └── pruning_v2.py
│
├── quantization/
│ ├── quantization_v1.py
│ └── quantization_eval.py
│
├── results/
│ ├── latency_results/ # screenshots or exported tables from report
│ ├── roi_results/ # ROI comparison images
│ └── other_figures/ # additional experiment visualizations
│
├── model.py
├── config_v1.py
└── README.md
```

## 🧪 Running Experiments

### Knowledge Distillation
```bash
python knowledge_distillation/distillation_v1.py --config config_v1.py
python knowledge_distillation/distillation_v2.py --config config_v1.py
```

### Quantization
```bash
python quantization/quantization_v1.py
python quantization/quantization_eval.py
```

### Pruning
```bash
python pruning/pruning_v1.py
python pruning/pruning_v2.py
```

## 📊 Results Summary

| Method                   | PLL Change | Latency Impact | Notes                |
|--------------------------|------------|----------------|----------------------|
| KD + INT8 Quantization   | –0.15%     | –0.14 ms       | Best trade-off       |
| Feature KD               | Slight gain| Minimal        | Strong representations|
| Basic KD                 | Minor drop | Slight gain    | Good baseline        |
| Pruning (proto)          | N/A        | N/A            | Not fully integrated |

## 🧭 Interpretation & Discussion

Our findings show that **knowledge distillation paired with INT8 quantization** provides the best balance of accuracy and efficiency for conversion prediction models operating in latency-constrained environments.

### Key takeaways:
- KD provides representational alignment, enabling a smaller student to approximate the teacher’s behavior.
- Quantization reduces computational cost while retaining stability at 8-bit precision.
- Combining both techniques yields a student model capable of meeting production requirements without sacrificing prediction quality.
- Positive ROI improvements suggest that computational optimizations can translate into meaningful business-level value for ad-serving systems.
- Early pruning prototypes indicate potential for additional compression once integrated with KD and quantization.

These results reinforce that carefully designed compression pipelines are viable alternatives to increasing model size, especially when real-time inference constraints dominate system design.

## 🔭 Future Work

- Structured and movement pruning  
- Quantization-aware training (QAT) for lower precision  
- Automated hyperparameter search  
- Serving-side benchmarking under high load  
- Full integration into continuous deployment pipelines  

## 📚 References
- Hinton et al., *Distilling the Knowledge in a Neural Network* (2015)  
- Romero et al., *FitNets: Hints for Thin Deep Nets* (2014)  
- Jacob et al., *Integer-Only Quantization for Neural Networks* (2017)  
- Krishnamoorthi, *Efficient Inference Quantization* (2018)  
- LeCun et al., *Optimal Brain Damage* (1989)  
- Courbariaux & Bengio, *BinaryConnect* (2015)  
- Project Report (HPML Final), Columbia University  

## 👥 Team
- Mahdi Saleh Tabesh  
- Alex Racapé  
- Rohan Singh  
- Kimberly Collins  

## 📄 License
MIT License
