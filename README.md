# Infrared_Image_Survey
# Infrared / Thermal Imaging Papers (Tone Mapping · Denoising · Super-Resolution · Enhancement)

> Columns: **| Method | Title | Paper Link | Code Link | Journal / Conference | Task | Model |**

---

## Table of Contents
- [Thermal Infrared Imaging / Tone Mapping](#thermal-infrared-imaging--tone-mapping)
- [Infrared Denoising](#infrared-denoising)
- [Infrared Denoising / NUC / Destriping](#infrared-denoising--nuc--destriping)
- [Infrared Super-Resolution](#infrared-super-resolution)
- [Infrared Joint Enhancement](#infrared-joint-enhancement)
- [Infrared Contrast Enhancement](#infrared-contrast-enhancement)


---


## Thermal Infrared Imaging / Tone Mapping

| Method | Title | Paper Link | Code Link | Journal / Conference | Task | Model |
|---|---|---|---|---|---|---|
| TCNet (ThermalChameleon) | Thermal Chameleon: Task-Adaptive Tone-mapping for Radiometric Thermal-Infrared Images | [arXiv](https://arxiv.org/abs/2410.18340) | [GitHub](https://github.com/donkeymouse/ThermalChameleon) | RA-L (2024) | Thermal Infrared Tone Mapping | CNN |
| TMIQA (Teutsch20) | An Evaluation of Objective Image Quality Assessment for Thermal Infrared Video Tone Mapping | [CVPRW](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Teutsch_An_Evaluation_of_Objective_Image_Quality_Assessment_for_Thermal_Infrared_Video_Tone_Mapping_CVPRW_2020_paper.html) | [GitHub](https://github.com/HensoldtOptronicsCV/ToneMappingIQA) | CVPRW (2020) | Thermal Infrared Tone Mapping | Metric |
| Fieldscale | Fieldscale: Locality-Aware Field-Based Adaptive Rescaling for Thermal Infrared Image | [arXiv](https://arxiv.org/abs/2405.15395) | [GitHub](https://github.com/HyeonJaeGil/fieldscale) | RA-L (2024) | Thermal Infrared Tone Mapping | Traditional |
| ShinCLAHE | Maximizing Self-Supervision from Thermal Image for Effective Self-Supervised Learning of Depth and Ego-Motion | [arXiv](https://arxiv.org/abs/2201.04387) | [GitHub](https://github.com/UkcheolShin/ThermalMonoDepth) | RA-L (2022) | Thermal Infrared Tone Mapping | CNN |


---



## Infrared Super-Resolution

| Method | Title | Paper Link | Code Link | Journal / Conference | Task | Model |
|---|---|---|---|---|---|---|
| UGSR (Gupta22) | Toward Unaligned Guided Thermal Super-Resolution | [DOI](https://doi.org/10.1109/TIP.2021.3130538) | [GitHub](https://github.com/honeygupta/UGSR) | IEEE TIP (2021) | Infrared Super-Resolution | CNN |
| TherISuRNet (Chudasama20) | TherISuRNet – A Computationally Efficient Thermal Image Super-Resolution Network | [IEEE](https://ieeexplore.ieee.org/document/9150703) | [GitHub](https://github.com/Vishal2188/TherISuRNet---A-Computationally-Efficient-Thermal-Image-Super-Resolution-Network) | CVPRW (2020) | Infrared Super-Resolution | CNN |
| CDN_MRF (He19) | Cascaded Deep Networks With Multiple Receptive Fields for Infrared Image Super-Resolution | [DOI](https://doi.org/10.1109/TCSVT.2018.2864777) | [GitHub](https://github.com/hezw2016/CDN_MRF) | IEEE TCSVT (2018) | Infrared Super-Resolution | CNN |
| TISR-Diffusion (Cortes24) | Exploring the Usage of Diffusion Models for Thermal Image Super-Resolution: A Generic, Uncertainty-Aware Approach for Guided and Non-Guided Schemes | [PDF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Cortes-Mendez_Exploring_the_Usage_of_Diffusion_Models_for_Thermal_Image_Super-resolution_CVPRW_2024_paper.pdf) | [GitHub](https://github.com/alcros33/ThermalSuperResolution) | CVPRW (2024) | Infrared Super-Resolution | Diffusion |
| CoRPLE | Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00391.pdf) | [GitHub](https://github.com/hey-it-s-me/CoRPLE) | ECCV (2024) | Infrared Super-Resolution | Hybrid |
| DifIISR | DifIISR: A Diffusion Model with Gradient Guidance for Infrared Image Super-Resolution | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_DifIISR_A_Diffusion_Model_with_Gradient_Guidance_for_Infrared_Image_CVPR_2025_paper.pdf) | [GitHub](https://github.com/zirui0625/DifIISR) | CVPR (2025) | Infrared Super-Resolution | Diffusion |
| ChasNet | Channel Split Convolutional Neural Network (ChaSNet) for Thermal Image Super-Resolution | [PDF](https://openaccess.thecvf.com/content/CVPR2021W/PBVS/papers/Prajapati_Channel_Split_Convolutional_Neural_Network_ChaSNet_for_Thermal_Image_Super-Resolution_CVPRW_2021_paper.pdf) | [GitHub](https://github.com/kalpeshjp89/ChasNet) | CVPRW (2021) | Infrared Super-Resolution | CNN |
| PSRGAN-TL (Huang21) | Infrared Image Super-Resolution via Transfer Learning and PSRGAN | [DOI](https://doi.org/10.1109/LSP.2021.3077801) | [GitHub](https://github.com/yongsongH/Infrared_Image_SR_PSRGAN) | IEEE SPL (2021) | Infrared Super-Resolution | CNN |
| DASR | DASR: Dual-Attention Transformer for Infrared Image Super-Resolution | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1350449523002955) | [GitHub](https://github.com/VDT-2048/DASR) | Infrared Physics & Technology (2023) | Infrared Super-Resolution | Transformer |
| MGNet-VGTSR | Thermal UAV Image Super-Resolution Guided by Multiple Visible Cues | [DOI](https://doi.org/10.1109/TGRS.2023.3234058) | [GitHub](https://github.com/mmic-lcl/Datasets-and-benchmark-code) | IEEE TGRS (2023) | Infrared Super-Resolution | CNN |
| HAT | Activating More Pixels in Image Super-Resolution Transformer | [arXiv](https://arxiv.org/abs/2205.04437) | [GitHub](https://github.com/XPixelGroup/HAT) | CVPR (2023) | Infrared Super-Resolution | Transformer |
| SwinIR | SwinIR: Image Restoration Using Swin Transformer | [arXiv](https://arxiv.org/abs/2108.10257) | [GitHub](https://github.com/JingyunLiang/SwinIR) | ICCVW (2021) | Infrared Super-Resolution | Transformer |
| FeMaSR | Real-World Blind Super-Resolution via Feature Matching with Implicit High-Resolution Priors | [arXiv](https://arxiv.org/abs/2202.13142) | [GitHub](https://github.com/chaofengc/FeMaSR) | ACM MM (2022) | Infrared Super-Resolution | Hybrid |
| TESR | Edge-Enhanced Infrared Image Super-Resolution Reconstruction Model Under Transformer | [PDF](https://www.nature.com/articles/s41598-024-66302-8.pdf) | — | Scientific Reports (Nature) (2024) | Infrared Super-Resolution | Transformer |

---

## Infrared Contrast Enhancement

| Method | Title | Paper Link | Code Link | Journal / Conference | Task | Model |
|---|---|---|---|---|---|---|
| IE-CGAN (Kuang19) | Single Infrared Image Enhancement Using a Deep Convolutional Neural Network | [DOI](https://doi.org/10.1016/j.neucom.2018.11.081) | [GitHub](https://github.com/Kuangxd/IE-CGAN) | Neurocomputing (2019) | Infrared Contrast Enhancement | CNN |
| DGIF-IR | Infrared Image Enhancement Algorithm Based on Detail Enhancement Guided Image Filtering | [Springer](https://link.springer.com/article/10.1007/s00371-022-02741-6) | — | The Visual Computer (2023) | Infrared Contrast Enhancement | Traditional |
| IAT | You Only Need 90K Parameters to Adapt Light: A Light Weight Transformer for Image Enhancement and Exposure Correction | [arXiv](https://arxiv.org/abs/2205.14871) | [GitHub](https://github.com/cuiziteng/Illumination-Adaptive-Transformer) | BMVC (2022) | Infrared Contrast Enhancement | Transformer |
| 2D-GAN | Dual Decoding Generative Adversarial Networks for Infrared Image Enhancement | [PDF](s41598-025-06538-0.pdf) | [GitHub](https://github.com/Adriannajl/2D-GAN) | Scientific Reports (Nature) (2025) | Infrared Contrast Enhancement | GAN |
| AICT-DMTH | Advanced Enhancement Technique for Infrared Images of Wind Turbine Blades Utilizing Adaptive Difference Multi-Scale Top-Hat Transformation | [PDF](https://www.nature.com/articles/s41598-024-66423-0.pdf) | — | Scientific Reports (Nature) (2024) | Infrared Contrast Enhancement | Traditional |

---

## Infrared Joint Enhancement

| Method | Title | Paper Link | Code Link | Journal / Conference | Task | Model |
|---|---|---|---|---|---|---|
| DeepIR (Saragadam21) | Thermal Image Processing via Physics-Inspired Deep Networks | [arXiv](https://arxiv.org/abs/2108.07973) | [GitHub](https://github.com/vishwa91/DeepIR) | ICCVW (2021) | Infrared Joint Enhancement | Hybrid |
| TEN | Thermal Image Enhancement Using Convolutional Neural Network | [IEEE](https://ieeexplore.ieee.org/document/7759059) | [GitHub](https://github.com/ninadakolekar/Thermal-Image-Enhancement) | IROS (2016) | Infrared Joint Enhancement | CNN |
| DDcGAN | DDcGAN: A Dual-Discriminator Conditional GAN for Multi-Resolution Image Fusion | [DOI](https://doi.org/10.1109/TIP.2020.2977573) | [GitHub](https://github.com/hanna-xu/DDcGAN) | IEEE TIP (2020) | Infrared Joint Enhancement | CNN |
| HKDnet | Heterogeneous Knowledge Distillation for Simultaneous Infrared-Visible Image Fusion and Super-Resolution | [DOI](https://doi.org/10.1109/TIM.2022.3149101) | [GitHub](https://github.com/firewaterfire/HKDnet) | IEEE TIM (2022) | Infrared Joint Enhancement | CNN |
| DEAL | DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_DEAL_Data-Efficient_Adversarial_Learning_for_High-Quality_Infrared_Imaging_CVPR_2025_paper.pdf) | [GitHub](https://github.com/LiuZhu-CV/DEAL) | CVPR (2025) | Infrared Joint Enhancement | Hybrid |
| Hidden2D-Turb | Revelation of Hidden 2D Atmospheric Turbulence Strength Fields from Turbulence Effects in Infrared Imaging | [Nature](https://www.nature.com/articles/s43588-023-00498-z) | — | Nature Computational Science (2023) | Infrared Joint Enhancement | Hybrid |
| TFDL | Joint Tone Mapping and Denoising of Thermal Infrared Images via Multi-Scale Retinex and Multi-Task Learning | [arXiv](https://arxiv.org/abs/2305.00691) | — | arXiv (2023) | Infrared Joint Enhancement | CNN |
| SD-EM | Simultaneous Destriping and Image Denoising Using a Nonparametric Model With the EM Algorithm | [DOI](https://doi.org/10.1109/TIP.2023.3239193) | [GitHub](https://github.com/slfff/Image-Destriping) | IEEE TIP (2023) | Infrared Joint Enhancement | Traditional |
| AirNet | All-in-One Image Restoration for Unknown Corruption | [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.pdf) | [GitHub](https://github.com/XLearning-SCU/2022-CVPR-AirNet) | CVPR (2022) | Infrared Joint Enhancement | CNN |
| DeDn-CNN | Thermal Fault Diagnosis of Complex Electrical Equipment Based on Infrared Image Recognition | [PDF](https://www.nature.com/articles/s41598-024-56142-x.pdf) | — | Scientific Reports (Nature) (2024) | Infrared Joint Enhancement | CNN + Traditional |

---
## Infrared Denoising / NUC / Destriping

| Method | Title | Paper Link | Code Link | Journal / Conference | Task | Model |
|---|---|---|---|---|---|---|
| ASCNet | ASCNet: Asymmetric Sampling Correction Network for Infrared Image Destriping | [arXiv](https://arxiv.org/abs/2401.15578) | [GitHub](https://github.com/xdFai/ASCNet) | IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2025 | Infrared NUC | CNN |
| 1D-GF (Cao16) | Effective Stripe Noise Removal for Low-Textured Infrared Images Based on 1-D Guided Filtering | [DOI](https://doi.org/10.1109/TCSVT.2015.2493443) | [GitHub](https://github.com/hezw2016/1D-GF) | IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2016 | Infrared NUC | Traditional |
| SemiCal-FPN | Fixed Pattern Noise Removal Based on a Semi-Calibration Method | [DOI](https://doi.org/10.1109/TPAMI.2023.3274826) | — | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023 | Infrared NUC | Optimization |
| MultiView-FPN (Barral24) | Fixed Pattern Noise Removal for Multi-View Single-Sensor Infrared Camera | [WACV](https://openaccess.thecvf.com/content/WACV2024/html/Barral_Fixed_Pattern_Noise_Removal_for_Multi-View_Single-Sensor_Infrared_Camera_WACV_2024_paper.html) | [GitHub](https://github.com/centreborelli/multiview-fpn) | WACV, 2024 | Infrared NUC | CNN |
| MIRE | Non-uniformity Correction of Infrared Images by Midway Equalization | [IPOL](https://www.ipol.im/pub/art/2012/glmt-mire/) | [GitHub](https://github.com/tguillemot/midway_equalization) | Image Processing On Line (IPOL), 2012 | Infrared NUC | Traditional |
| DestripeCycleGAN | DestripeCycleGAN: Stripe Simulation CycleGAN for Unsupervised Infrared Image Destriping | [arXiv](https://arxiv.org/abs/2402.09101) | [GitHub](https://github.com/xdFai/DestripeCycleGAN) | IEEE Transactions on Instrumentation and Measurement (TIM), 2024 | Infrared NUC | GAN |
| DMRN | Infrared Aerothermal Nonuniformity Correction via Deep Multiscale Residual Network | [DOI](https://doi.org/10.1109/LGRS.2019.2893519) | — | IEEE Geoscience and Remote Sensing Letters (GRSL), 2019 | Infrared NUC | CNN |
| CNN-FPNR | Fixed Pattern Noise Reduction for Infrared Images Based on Cascade Residual Attention CNN | [arXiv](https://arxiv.org/abs/1910.09858) | — | Neurocomputing, 2020 | Infrared NUC | CNN |
| TV-DIP | Thermal Imaging Spatial Noise Removal via Deep Image Prior and Step-Variable Total Variation Regularization | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1350449523001158) | [GitHub](https://github.com/brunolinux/TV-DIP) | Infrared Physics & Technology, 2023 | Infrared NUC | Hybrid |
| Cheby-Fit | Multi-Scale Thermal Radiation Effects Correction via Fast Surface Fitting with Chebyshev Polynomials | [DOI](https://doi.org/10.1364/AO.465157) | — | Applied Optics, 2022 | Infrared NUC | Polynomial |
| AHBC | Thermal Radiation Bias Correction for Infrared Images Using Huber Function-Based Loss | [DOI](https://doi.org/10.1109/TGRS.2024.3370966) | — | IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2024 | Infrared NUC | Optimization |
| DLS-NUC (He18) | Single-Image-Based Nonuniformity Correction of Uncooled Long-Wave Infrared Detectors: A Deep-Learning Approach | [DOI](https://doi.org/10.1364/AO.57.00D155) | [GitHub](https://github.com/hezw2016/DLS-NUC) | Applied Optics, 2018 | Infrared NUC | CNN |
| SNRWDNN | Wavelet Deep Neural Network for Stripe Noise Removal | [IEEE](https://ieeexplore.ieee.org/document/8678750) | [GitHub](https://github.com/jtguan/Wavelet-Deep-Neural-Network-for-Stripe-Noise-Removal) | IEEE Access, 2019 | Infrared NUC | CNN |
| IDTransformer | IDTransformer: Infrared Image Denoising Method Based on Convolutional Transposed Self-Attention | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1110016824011256) | [GitHub](https://github.com/szw811/IDTransformer) | Alexandria Engineering Journal (AEJ), 2025 | Infrared Image Denoising | Transformer |
| MLFAN | Infrared Image Denoising via Adversarial Learning with Multi-Level Feature Attention Network | [DOI](https://doi.org/10.1016/j.infrared.2022.104527) | — | Infrared Physics & Technology, 2023 | Infrared Image Denoising | CNN |
| SMNet | Infrared Thermal Image Denoising with Symmetric Multi-Scale Sampling Network | [DOI](https://doi.org/10.1016/j.infrared.2023.104909) | — | Infrared Physics & Technology, 2023 | Infrared Image Denoising | CNN |
| MIVDN | Exploring Video Denoising in Thermal Infrared Imaging: Physics-Inspired Noise Generator, Dataset, and Model | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38652635/) | — | IEEE Transactions on Image Processing (TIP), 2024 | Infrared Image Denoising | CNN |
| UTV | Toward Optimal Destriping of MODIS Data Using a Unidirectional Variational Model | [DOI](https://doi.org/10.1109/TGRS.2011.2119399) | — | IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2011 | Infrared Image Denoising | Optimization |
| LRSID | Remote Sensing Image Stripe Noise Removal: From Image Decomposition Perspective | [DOI](https://doi.org/10.1109/TGRS.2016.2594080) | — | IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2016 | Infrared Image Denoising | Optimization |
| TSWEU | Toward Universal Stripe Removal via Wavelet-Based Deep Convolutional Neural Network | [DOI](https://doi.org/10.1109/TGRS.2019.2957153) | — | IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2020 | Infrared Image Denoising | CNN |


---









