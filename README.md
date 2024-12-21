# Optimization for Large Language Models (OPT4LLM)

I currently focus on optimization for large language models (OPT4LLM) including
- [Surveys](#Surveys)
- [Pruning](#Pruning)
- [Quantization](#Quantization)
- [Fine-Tuning](#Fine-Tuning)
  
<strong> Last Update: 2024/12/20 </strong>



<a name="Surveys" />

### Surveys
- [2024] A Survey of Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2303.18223)] [[Code](https://github.com/RUCAIBox/LLMSurvey)]
- [2024] A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations, IEEE TPAMI  [[Paper](https://ieeexplore.ieee.org/abstract/document/10643325/)] [[Code](https://github.com/hrcheng1066/awesome-pruning)]
- [2024] A Systematic Survey on Large Language Models for Algorithm Design, arXiv [[Paper](https://arxiv.org/abs/2410.14716)]
- [2024] Efficient Large Language Models: A Survey, TMLR [[Paper](https://arxiv.org/abs/2312.03863)] [[Code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)]
- [2024] A Survey of Low-Bit Large Language Models: Basics, Systems, and Algorithms, arXiv [[Paper](https://arxiv.org/abs/2409.16694)]
- [2024] A Survey of Mamba, arXiv [[Paper](https://arxiv.org/abs/2408.01129)] 
- [2024] Small Language Models: Survey, Measurements, and Insights, arXiv [[Paper](https://arxiv.org/abs/2409.15790)]
- [2023] A Survey on Multimodal Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2306.13549)]    
- [2022] Compression of Deep Learning Models for Text: A Survey, ACM TKDD [[Paper](https://dl.acm.org/doi/full/10.1145/3487045)]




 
<a name="Pruning" />

### Pruning


- [2024] Two Sparse Matrices are Better than One: Sparsifying Neural Networks with Double Sparse Factorization, arXiv [[Paper](https://arxiv.org/abs/2409.18850)]  [[Code](https://github.com/usamec/double_sparse)]
- [2024] Fast and Effective Weight Update for Pruned Large Language Models, TMLR [[Paper](https://openreview.net/forum?id=1hcpXd9Jir)] [[Code](https://github.com/fmfi-compbio/admm-pruning)]
- [2024] A Simple and Effective Pruning Approach for Large Language Models, ICLR [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]
- [2023] LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation, ICML [[Paper](https://proceedings.mlr.press/v202/li23ap.html)] [[Python](https://github.com/yxli2123/LoSparse)]
- [2023] SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot, ICML [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- [2020] Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression, CVPR [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Group_Sparsity_The_Hinge_Between_Filter_Pruning_and_Decomposition_for_CVPR_2020_paper.html)] [[Code](https://github.com/ofsoundof/group_sparsity)]
- [2017] Channel Pruning for Accelerating Very Deep Neural Networks, ICCV [[Paper](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html)]


- Compact Language Models via Pruning and Knowledge Distillation, <ins>arXiv, 2024</ins> [[Paper](https://www.arxiv.org/abs/2407.14679)] 
- A deeper look at depth pruning of LLMs, <ins>arXiv, 2024</ins> [[Paper](https://www.arxiv.org/abs/2407.16286)] 
- Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.20541)] 
- Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models, <ins>ICLR, 2024</ins> [[Paper](https://openreview.net/forum?id=Tr0lPx9woF)] 
- BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.16880)] 
- ShortGPT: Layers in Large Language Models are More Redundant Than You Expect, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.03853)] 
- NutePrune: Efficient Progressive Pruning with Numerous Teachers for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.09773)] 
- SliceGPT: Compress Large Language Models by Deleting Rows and Columns, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2401.15024)] [[Code](https://github.com/microsoft/TransformerCompression?utm_source=catalyzex.com)]
- LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.18356)]
- LLM-Pruner: On the Structural Pruning of Large Language Models, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.11627)] [[Code](https://github.com/horseee/LLM-Pruner)]
- Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning, <ins> NeurIPS-ENLSP, 2023</ins> [[Paper](https://arxiv.org/abs/2310.06694)] [[Code](https://github.com/princeton-nlp/LLM-Shearing)]
- LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning, <ins>arXiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2305.18403)]
- LoRAP: Transformer Sub-Layers Deserve Differentiated Structured Compression for Large Language Models, <ins>ICML, 2024</ins> [[Paper](https://arxiv.org/abs/2404.09695)][[Code](https://github.com/lihuang258/LoRAP)]


- MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models, <ins>NIPS, 2024</ins> [[Paper](https://arxiv.org/abs/2409.17481)] 
- Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2310.08915)] 
- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- A Simple and Effective Pruning Approach for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]

- 

<a name="Quantization" />

### Quantization

- [2024] OneBit: Towards Extremely Low-bit Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2402.11295)] 

- I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.17849)] 
- IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.01241)] 
- OmniQuant: OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
- OneBit: Towards Extremely Low-bit Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.11295)]
- GPTQ: Accurate Quantization for Generative Pre-trained Transformers, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=tcbBPnfwxS)] [[Code](https://github.com/IST-DASLab/gptq)]
- QuIP: 2-Bit Quantization of Large Language Models With Guarantees, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.13304)] [[Code](https://github.com/jerry-chee/QuIP)]
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.00978)] [[Code](https://github.com/mit-han-lab/llm-awq)]
- OWQ: Lessons Learned from Activation Outliers for Weight Quantization in Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.02272)] [[Code](https://github.com/xvyaward/owq)]
- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2306.03078)] [[Code](https://github.com/Vahe1994/SpQR)]
- FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs, <ins>NeurIPS-ENLSP, 2023</ins> [[Paper](https://arxiv.org/abs/2308.09723)]
- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=dXiGWqBoxaD)] [[Code](https://github.com/TimDettmers/bitsandbytes)]
- Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2208.11580)] [[Code](https://github.com/IST-DASLab/OBC)]
- QuantEase: Optimization-based Quantization for Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.01885)] [[Code](https://github.com/linkedin/QuantEase)]


- Rotation and Permutation for Advanced Outlier Management and Efficient Quantization of LLMs, <ins>NeurIPS, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] 
- OmniQuant: OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
- Intriguing Properties of Quantization at Scale, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.19268)]
- ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2303.08302)] [[Code](https://github.com/microsoft/DeepSpeed)]
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats, <ins>NeurIPS-ENLSP, 2023</ins> [[Paper](https://arxiv.org/abs/2307.09782)] [[Code](https://github.com/microsoft/DeepSpeed)]
- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization, <ins>ISCA, 2023</ins> [[Paper](https://arxiv.org/abs/2304.07493)] [[Code](https://github.com/clevercool/ANT-Quantization)]
- RPTQ: Reorder-based Post-training Quantization for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.01089)] [[Code](https://github.com/hahnyuan/RPTQ4LLM)]
- Outlier Suppression+: Accurate Quantization of Large Language Models by Equivalent and Optimal Shifting and Scaling, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.09145)] [[Code](https://github.com/ModelTC/Outlier_Suppression_Plus)]
- QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.08041)]
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2211.10438)] [[Code](https://github.com/mit-han-lab/smoothquant)]
- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers, <ins>NeurIPS, 2022</ins> [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html)]



 - Evaluating Quantized Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.18158)]


- The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.17764)]
- FP8-LM: Training FP8 Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.18313)]
- Training and inference of large language models using 8-bit floating point, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.17224)]
- BitNet: Scaling 1-bit Transformers for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.11453)]
- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17888)] [[Code](https://github.com/facebookresearch/LLM-QAThttps://github.com/facebookresearch/LLM-QAT)]
- Compression of Generative Pre-trained Language Models via Quantization, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.331.pdf)]

- 

### Fine-Tuning
- [2024] Adaptive Principal Components Allocation with the l2,g-regularized Gaussian Graphical Model for Efficient Fine-Tuning Large Models, arXiv [[Paper](https://arxiv.org/abs/2412.08592)] [[Code](https://github.com/jzheng20/Course_projects.git)]
- [2023] Visual Instruction Tuning, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html)]
- [2023] QLoRA: Efficient Finetuning of Quantized LLMs, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)]
- [2022] LoRA: Low-Rank Adaptation of Large Language Models, ICLR [[Paper](https://openreview.net/forum?id=nZeVKeeFYf9)] [[Code](https://github.com/microsoft/LoRA)]



- HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning, <ins>NeurIPS, 2024</ins> [[Paper](https://arxiv.org/pdf/2404.19245)]
- LOFIT: Localized Fine-tuning on LLM Representations, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.01563)]
- Mixture-of-Subspaces in Low-Rank Adaptation, <ins>Arxiv, 2024</ins> [[Paper](https://arxiv.org/pdf/2406.11909)] [[Code](https://github.com/wutaiqiang/MoSLoRA)]
- MEFT: Memory-Efficient Fine-Tuning through Sparse Adapter, <ins>ACL, 2024</ins> [[Paper](https://arxiv.org/html/2406.04984v1)]
- LoRA Meets Dropout under a Unified Framework, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.00812)]
- STAR: Constraint LoRA with Dynamic Active Learning for Data-Efficient Fine-Tuning of Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.01165)]
- LoRA+: Efficient Low Rank Adaptation of Large Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.12354)]
- LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.03303)]
- LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.13269)] [[Code](https://github.com/sail-sg/lorahub)]
- LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.12307)] [[Code](https://github.com/dvlab-research/LongLoRA)]
- Multi-Head Adapter Routing for Cross-Task Generalization, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2211.03831)] [[Code](https://github.com/microsoft/mttl)]
- Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/pdf/2303.10512)] 
- DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low Rank Adaptation, <ins>EACL, 2023</ins> [[Paper](https://aclanthology.org/2023.eacl-main.239/)] [[Code](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)]
- Tied-Lora: Enhacing Parameter Efficiency of LoRA with Weight Tying, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.09578)]
- LoRA: Low-Rank Adaptation of Large Language Models, <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=nZeVKeeFYf9)] [[Code](https://github.com/microsoft/LoRA)]

- A Study of Optimizations for Fine-tuning Large Language Models, <ins>arXiv, 2024/ins> [[Paper](https://arxiv.org/abs/2406.02290)] 
- Sparse Matrix in Large Language Model Fine-tuning, <ins>arXiv, 2024/ins> [[Paper](https://arxiv.org/pdf/2405.15525)] 
- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection, <ins>arXiv, 2024/ins> [[Paper](https://arxiv.org/abs/2403.03507)] 
- ReFT: Representation Finetuning for Language Models, <ins>arXiv, 2024/ins> [[Paper](https://arxiv.org/abs/2404.03592)] 
- LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning, <ins>arXiv, 2024/ins> [[Paper](https://arxiv.org/abs/2403.17919)] 
- BitDelta: Your Fine-Tune May Only Be Worth One Bit, <ins>arXiv, 2024/ins> [[Paper](https://arxiv.org/abs/2402.10193)] 
- Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.15265)] [[Code](https://github.com/zirui-ray-liu/WTACRS)]
- Memory-Efficient Selective Fine-Tuning, <ins>ICML Workshop, 2023</ins> [[Paper](https://openreview.net/forum?id=zaNbLceVwm)]
- Full Parameter Fine-tuning for Large Language Models with Limited Resources, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09782)] [[Code](https://github.com/OpenLMLab/LOMO)]
- Fine-Tuning Language Models with Just Forward Passes, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17333)] [[Code](https://github.com/princeton-nlp/MeZO)]
- Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14152)]
- LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.08659)] [[Code](https://github.com/yxli2123/LoftQ)]
- QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.14717 )] [[Code](https://github.com/yuhuixu1993/qa-lora)]
- QLoRA: Efficient Finetuning of Quantized LLMs, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14314)] [[Code1](https://github.com/artidoro/qlora)] [[Code2](https://github.com/TimDettmers/bitsandbytes)]

