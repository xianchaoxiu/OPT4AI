# Optimization for Artificial Intelligence (OPT4AI)

I currently focus on optimization for artificial intelligence (OPT4AI), especially large language models, including

- [Surveys](#Surveys)
- [Pruning](#Pruning)
- [Quantization](#Quantization)
- [Binarization](#Binarization)
- [Fine-Tuning](#Fine-Tuning)
- [Resources](#Resources)
    
<strong> Last Update: 2025/01/23 </strong>



<a name="Surveys" />

## Surveys
- [2025] Logical Reasoning in Large Language Models: A Survey, arXiv [[Paper](https://arxiv.org/abs/2502.09100)]
- [2025] Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG, arXiv [[Paper](https://arxiv.org/abs/2501.09136)]
- [2025] Continual Learning With Knowledge Distillation: A Survey, IEEE TNNLS [[Paper](https://ieeexplore.ieee.org/document/10721446)]
- [2025] Low-Rank Adaptation for Foundation Models: A Comprehensive Review, arXiv [[Paper](https://arxiv.org/pdf/2501.00365)]
- [2024] A Study of Optimizations for Fine-tuning Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2406.02290)] 
- [2024] A Survey of Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2303.18223)] [[Code](https://github.com/RUCAIBox/LLMSurvey)]
- [2024] Prompt Compression for Large Language Models: A Survey, arXiv [[Paper](https://arxiv.org/abs/2410.12388)]
- [2024] Efficient Large Language Models: A Survey, TMLR [[Paper](https://arxiv.org/abs/2312.03863)] [[Code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)]
- [2024] A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations, IEEE TPAMI  [[Paper](https://ieeexplore.ieee.org/abstract/document/10643325/)] [[Code](https://github.com/hrcheng1066/awesome-pruning)]
- [2024] A Systematic Survey on Large Language Models for Algorithm Design, arXiv [[Paper](https://arxiv.org/abs/2410.14716)]
- [2024] A Survey of Low-Bit Large Language Models: Basics, Systems, and Algorithms, arXiv [[Paper](https://arxiv.org/abs/2409.16694)]
- [2024] Small Language Models: Survey, Measurements, and Insights, arXiv [[Paper](https://arxiv.org/abs/2409.15790)]  [[Code](https://github.com/UbiquitousLearning/SLM_Survey)]
- [2023] A Survey on Multimodal Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2306.13549)]    
- [2022] Compression of Deep Learning Models for Text: A Survey, ACM TKDD [[Paper](https://dl.acm.org/doi/full/10.1145/3487045)]
- [2020] Optimization for Deep Learning: An Overview, JORC [[Paper](https://link.springer.com/article/10.1007/s40305-020-00309-6)]
- [2020] Binary Neural Networks: A Survey, PR  [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320320300856)]
 
<a name="Pruning" />

## Pruning
- [2025] FASP: Fast and Accurate Structured Pruning of Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2501.09412)]
- [2024] Two Sparse Matrices are Better than One: Sparsifying Neural Networks with Double Sparse Factorization, arXiv [[Paper](https://arxiv.org/abs/2409.18850)]  [[Code](https://github.com/usamec/double_sparse)]
- [2024] Fast and Effective Weight Update for Pruned Large Language Models, TMLR [[Paper](https://openreview.net/forum?id=1hcpXd9Jir)] [[Code](https://github.com/fmfi-compbio/admm-pruning)]
- [2024] A Simple and Effective Pruning Approach for Large Language Models, ICLR [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]
- [2024] Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models, ICLR [[Paper](https://openreview.net/forum?id=Tr0lPx9woF)] 
- [2024] BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation, arXiv [[Paper](https://arxiv.org/abs/2402.16880)]
- [2024] MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models, NeurIPS [[Paper](https://arxiv.org/abs/2409.17481)] 
- [2024] Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs, ICLR [[Paper](https://arxiv.org/abs/2310.08915)] 
- [2023] LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning, arXiv [[Paper](https://doi.org/10.48550/arXiv.2305.18403)]
- [2023] LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation, ICML [[Paper](https://proceedings.mlr.press/v202/li23ap.html)] [[Code](https://github.com/yxli2123/LoSparse)]
- [2023] SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot, ICML [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- [2020] Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression, CVPR [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Group_Sparsity_The_Hinge_Between_Filter_Pruning_and_Decomposition_for_CVPR_2020_paper.html)] [[Code](https://github.com/ofsoundof/group_sparsity)]



<a name="Quantization" />

## Quantization 

- [2024] OneBit: Towards Extremely Low-bit Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2402.11295)] 
- [2024] I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2405.17849)] 
- [2024] Evaluating Quantized Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2402.18158)]
- [2024] The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits, arXiv [[Paper](https://arxiv.org/abs/2402.17764)]
- [2023] Post-training Quantization for Neural Networks with Provable Guarantees, SIMODS [[Paper](https://epubs.siam.org/doi/abs/10.1137/22M1511709)]
- [2023] Training and inference of large language models using 8-bit floating point, arXiv [[Paper](https://arxiv.org/abs/2309.17224)]
- [2023] BitNet: Scaling 1-bit Transformers for Large Language Models, arXiv[[Paper](https://arxiv.org/abs/2310.11453)]
- [2023] LLM-QAT: Data-Free Quantization Aware Training for Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2305.17888)] [[Code](https://github.com/facebookresearch/LLM-QAThttps://github.com/facebookresearch/LLM-QAT)]
- [2023] GPTQ: Accurate Quantization for Generative Pre-trained Transformers, ICLR [[Paper](https://openreview.net/forum?id=tcbBPnfwxS)] [[Code](https://github.com/IST-DASLab/gptq)]
- [2023] QuIP: 2-Bit Quantization of Large Language Models With Guarantees, arXiv [[Paper](https://arxiv.org/abs/2307.13304)] [[Code](https://github.com/jerry-chee/QuIP)]
- [2023] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration,  arXiv  [[Paper](https://arxiv.org/abs/2306.00978)] [[Code](https://github.com/mit-han-lab/llm-awq)]
- [2023] OWQ: Lessons Learned from Activation Outliers for Weight Quantization in Large Language Models, arXiv </ins> [[Paper](https://arxiv.org/abs/2306.02272)] [[Code](https://github.com/xvyaward/owq)]
- [2023] SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression, arXiv [[Paper](https://arxiv.org/pdf/2306.03078)] [[Code](https://github.com/Vahe1994/SpQR)]
- [2023] QuantEase: Optimization-based Quantization for Language Models, arXiv [[Paper](https://arxiv.org/abs/2309.01885)] [[Code](https://github.com/linkedin/QuantEase)]
- [2023] ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation, arXiv [[Paper](https://arxiv.org/abs/2303.08302)] [[Code](https://github.com/microsoft/DeepSpeed)]
- [2023] QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2310.08041)]
- [2023] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, ICML [[Paper](https://arxiv.org/abs/2211.10438)] [[Code](https://github.com/mit-han-lab/smoothquant)]
- [2022] ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html)]
- [2022] Compression of Generative Pre-trained Language Models via Quantization, ACL [[Paper](https://aclanthology.org/2022.acl-long.331.pdf)]



<a name="Binarization" />

## Binarization

- [2024] Scalable Binary Neural Network Applications in Oblivious Inference, ACM TECS [[Paper](https://dl.acm.org/doi/full/10.1145/3607192)]
- [2022] Structured Binary Neural Networks for Image Recognition, IJCV [[Paper](https://link.springer.com/article/10.1007/s11263-022-01638-0)]
- [2022] Data-Adaptive Binary Neural Networks for Efficient Object Detection and Recognition, PRL[[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865521004438)]
- [2021] Learning Frequency Domain Approximation for Binary Neural Networks, NeurIPS [[Paper](https://proceedings.neurips.cc/paper/2021/hash/d645920e395fedad7bbbed0eca3fe2e0-Abstract.html)]
- [2021] Layer-Wise Searching for 1-Bit Detectors, CVPR [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_Layer-Wise_Searching_for_1-Bit_Detectors_CVPR_2021_paper.html)]
- [2020] Rotated Binary Neural Network, NeurIPS [[Paper](https://proceedings.neurips.cc/paper/2020/hash/53c5b2affa12eed84dfec9bfd83550b1-Abstract.html)]
- [2018] Regularized Binary Network Training, arXiv [[Paper](https://arxiv.org/abs/1812.11800)]
- [2018] Enhancing the Performance of 1-bit CNNs with Improved Representational Capability and Advanced Training Algorithm, ECCV [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.pdf)]
- [2018] Circulant Binary Convolutional Networks: Enhancing the Performance of 1-bit DCNNs with Circulant Back Propagation, CVPR [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Circulant_Binary_Convolutional_Networks_Enhancing_the_Performance_of_1-Bit_DCNNs_CVPR_2019_paper.html)]
- [2017] How to Train a Compact Binary Neural Network with High Accuracy? AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10862)]
- [2015] BinaryConnect: Training Deep Neural Networks with Binary Weights During Propagations, NIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/3e15cc11f979ed25912dff5b0669f2cd-Paper.pdf)]



<a name="Fine-Tuning" />

## Fine-Tuning

- [2024] Adaptive Principal Components Allocation with the L2,g-regularized Gaussian Graphical Model for Efficient Fine-Tuning Large Models, arXiv [[Paper](https://arxiv.org/abs/2412.08592)] [[Code](https://github.com/jzheng20/Course_projects.git)]
- [2024] PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2404.02948)] 
- [2024] Sparse Matrix in Large Language Model Fine-tuning, arXiv [[Paper](https://arxiv.org/pdf/2405.15525)] 
- [2024] GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection, arXiv [[Paper](https://arxiv.org/abs/2403.03507)] 
- [2024] ReFT: Representation Finetuning for Language Models, arXiv[[Paper](https://arxiv.org/abs/2404.03592)] 
- [2024] LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning, arXiv, [[Paper](https://arxiv.org/abs/2403.17919)] 
- [2024] BitDelta: Your Fine-Tune May Only Be Worth One Bit, arXiv [[Paper](https://arxiv.org/abs/2402.10193)] 
- [2024] Mixture-of-Subspaces in Low-Rank Adaptation, Arxiv [[Paper](https://arxiv.org/pdf/2406.11909)] [[Code](https://github.com/wutaiqiang/MoSLoRA)]
- [2024] MEFT: Memory-Efficient Fine-Tuning through Sparse Adapter, ACL [[Paper](https://arxiv.org/html/2406.04984v1)]
- [2024] LoRA Meets Dropout under a Unified Framework, arXiv [[Paper](https://arxiv.org/abs/2403.00812)]
- [2024] LoRA+: Efficient Low Rank Adaptation of Large Models, arXiv [[Paper](https://arxiv.org/abs/2402.12354)]
- [2023] AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning, ICLR  [[Paper](https://arxiv.org/abs/2303.10512)]
- [2023] Visual Instruction Tuning, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html)]
- [2023] QLoRA: Efficient Finetuning of Quantized LLMs, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)]
- [2023] LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning, arXiv [[Paper](https://arxiv.org/abs/2308.03303)]
- [2023] LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition, arXiv [[Paper](https://arxiv.org/abs/2307.13269)] [[Code](https://github.com/sail-sg/lorahub)]
- [2023] LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2309.12307)] [[Code](https://github.com/dvlab-research/LongLoRA)]
- [2023] DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low Rank Adaptation, EACL [[Paper](https://aclanthology.org/2023.eacl-main.239/)] [[Code](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)]
- [2023] Tied-Lora: Enhacing Parameter Efficiency of LoRA with Weight Tying, arXiv [[Paper](https://arxiv.org/abs/2311.09578)]
- [2023] LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2310.08659)] [[Code](https://github.com/yxli2123/LoftQ)]
- [2023] QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models, arXiv[[Paper](https://arxiv.org/abs/2309.14717 )] [[Code](https://github.com/yuhuixu1993/qa-lora)]
- [2023] QLoRA: Efficient Finetuning of Quantized LLMs, NeurIPS [[Paper](https://arxiv.org/abs/2305.14314)] [[Code](https://github.com/artidoro/qlora)] 
- [2022] LoRA: Low-Rank Adaptation of Large Language Models, ICLR [[Paper](https://openreview.net/forum?id=nZeVKeeFYf9)] [[Code](https://github.com/microsoft/LoRA)]




<a name="Resources" />

## Resources
- Large Language Model in Action [[Link](https://wangwei1237.github.io/LLM_in_Action/)]
- Chinese LLM [[Link](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)]
- LLM Action [[Link](https://github.com/liguodongiot/llm-action)]
- Efficient LLMs [[Link](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)]
- LLM Prune [[Link](https://github.com/pprp/Awesome-LLM-Prune)]
- Vision Mamba Models [[Link](https://github.com/Ruixxxx/Awesome-Vision-Mamba-Models)]
- Transformers [[Link](https://arxiv.org/abs/2406.09246)]
- GenAI Agents [[Link](https://github.com/NirDiamant/GenAI_Agents)]

