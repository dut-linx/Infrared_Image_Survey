# Infrared_Image_Survey
# Infrared / Thermal Imaging Papers (Tone Mapping · Denoising · Super-Resolution · Translation)

> Columns: **简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用（BibTeX）**

---

## 目录
- [Tone Mapping & IQA](#tone-mapping--iqa)
- [Tone Mapping + Denoising](#tone-mapping--denoising)
- [Infrared / Thermal Super-Resolution](#infrared--thermal-super-resolution)
- [Infrared Image Enhancement](#infrared-image-enhancement)
- [Infrared Denoising / NUC / Destriping](#infrared-denoising--nuc--destriping)
- [Visible → Infrared / Thermal Translation](#visible--infrared--thermal-translation)
- [Fusion + SR](#fusion--sr)
- [BibTeX](#bibtex)

---

## Tone Mapping & IQA

| 简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用 |
|---|---|---|---|---|---|---|
| TMIQA (Teutsch20) | An Evaluation of Objective Image Quality Assessment for Thermal Infrared Video Tone Mapping | [PDF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w6/Teutsch_An_Evaluation_of_Objective_Image_Quality_Assessment_for_Thermal_Infrared_CVPRW_2020_paper.pdf) | [GitHub](https://github.com/HensoldtOptronicsCV/ToneMappingIQA) | CVPRW (2020) | 红外视频的色调映射图像质量评估方法 | [BibTeX](#teutschcvprw2020) |
| TCNet (ThermalChameleon) | Thermal Chameleon: Task-Adaptive Tone-mapping for Radiometric Thermal-Infrared images | [arXiv](https://arxiv.org/abs/2410.18340) | [GitHub](https://github.com/donkeymouse/ThermalChameleon) | IEEE Robotics and Automation Letters (RA-L) (2024) | 红外图像 ToneMapping | [BibTeX](#dglee-2024-tcnet) |

---

## Tone Mapping + Denoising

| 简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用 |
|---|---|---|---|---|---|---|
| TFDL | Joint tone mapping and denoising of thermal infrared images via multi-scale Retinex and multi-task learning | [arXiv](https://arxiv.org/abs/2305.00691) | [GitHub](https://github.com/hulitaotom/Joint-Multi-Scale-Tone-Mapping-and-Denoising-for-HDR-Image-Enhancement) | WACVW (2022) | 红外图像 ToneMapping + 去噪 | [BibTeX](#wacvw9707563) |

---

## Infrared / Thermal Super-Resolution

| 简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用 |
|---|---|---|---|---|---|---|
| DifIISR (Li25) | DifIISR: Diffusion Model with Gradient Guidance for Infrared Image Super-Resolution | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_DifIISR_A_Diffusion_Model_with_Gradient_Guidance_for_Infrared_Image_CVPR_2025_paper.pdf) | [GitHub](https://github.com/zirui0625/DifIISR) | CVPR (2025) | 红外图像超分辨率重建 | [BibTeX](#li2025difiisr) |
| UGSR (Gupta22) | Toward Unaligned Guided Thermal Super-Resolution | [DOI](https://doi.org/10.1109/TIP.2021.3130538) | [GitHub](https://github.com/honeygupta/UGSR) | IEEE Transactions on Image Processing (TIP) (2022) | 红外图像超分辨率重建 | [BibTeX](#gupta2021toward) |
| TherISuRNet (Chudasama20) | TherISuRNet - A Computationally Efficient Thermal Image Super-Resolution Network | [PDF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w6/Chudasama_TherISuRNet_-_A_Computationally_Efficient_Thermal_Image_Super-Resolution_Network_CVPRW_2020_paper.pdf) | [GitHub](https://github.com/Vishal2188/TherISuRNet---A-Computationally-Efficient-Thermal-Image-Super-Resolution-Network) | CVPRW (PBVS) (2020) | 红外图像超分辨率重建 | [BibTeX](#cvprw9150703) |
| CDN_MRF (He19) | Cascaded Deep Networks With Multiple Receptive Fields for Infrared Image Super-Resolution | [PDF](https://research.utwente.nl/files/481835696/Cascaded_Deep_Networks_With_Multiple_Receptive_Fields_for_Infrared_Image_Super-Resolution.pdf) | [GitHub](https://github.com/hezw2016/CDN_MRF) | IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) (2019) | 红外图像超分辨率重建 | [BibTeX](#tcsvt8432397) |
| PSRGAN-TL (Huang21) | Infrared Image Super-Resolution via Transfer Learning and PSRGAN | [DOI](https://doi.org/10.1109/LSP.2021.3077801) | [GitHub](https://github.com/yongsongH/Infrared_Image_SR_PSRGAN) | IEEE Signal Processing Letters (SPL) (2021) | 红外图像超分辨率重建 | [BibTeX](#spl9424970) |
| TISR-Diffusion (Cortes24) | Exploring the usage of diffusion models for thermal image super-resolution: a generic, uncertainty-aware approach for guided and non-guided schemes | [PDF](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Cortes-Mendez_Exploring_the_Usage_of_Diffusion_Models_for_Thermal_Image_Super-resolution_CVPRW_2024_paper.pdf) | [GitHub](https://github.com/alcros33/ThermalSuperResolution) | CVPRW (PBVS) (2024) | 红外图像超分辨率重建 | [BibTeX](#cvprw10678449) |

---

## Infrared Image Enhancement

| 简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用 |
|---|---|---|---|---|---|---|
| IE-CGAN (Kuang19) | Single infrared image enhancement using a deep convolutional neural network | [DOI](https://dl.acm.org/doi/10.1016/j.neucom.2018.11.081) | [GitHub](https://github.com/Kuangxd/IE-CGAN) | Neurocomputing (2019) | 红外图像对比度增强 | [BibTeX](#neucom2019-iecgan) |
| TEN (Choi16) | Thermal Image Enhancement using Convolutional Neural Network | [IEEE](https://ieeexplore.ieee.org/document/7759059) | [GitHub](https://github.com/ninadakolekar/Thermal-Image-Enhancement?tab=readme-ov-file) | IROS (2016) | 热红外图像增强（CNN） | [BibTeX](#iros7759059) |

---

## Infrared Denoising / NUC / Destriping

| 简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用 |
|---|---|---|---|---|---|---|
| IDTransformer (Shen25) | IDTransformer: Infrared image denoising method based on convolutional transposed self-attention | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1110016824011256) | [GitHub](https://github.com/szw811/IDTransformer) | Alexandria Engineering Journal (AEJ) (2025) | 红外图像去噪 | [BibTeX](#a ej-shen2025) |
| MultiView-FPN (Barral24) | Fixed Pattern Noise Removal for Multi-View Single-Sensor Infrared Camera | [PDF](https://openaccess.thecvf.com/content/WACV2024/papers/Barral_Fixed_Pattern_Noise_Removal_for_Multi-View_Single-Sensor_Infrared_Camera_WACV_2024_paper.pdf) | [GitHub](https://github.com/centreborelli/multiview-fpn) | WACV (2024) | 红外图像去噪（FPN） | [BibTeX](#barral_2024_wacv) |
| DLS-NUC (He18) | Single-image-based nonuniformity correction of uncooled long-wave infrared detectors: a deep-learning approach | [Optica](https://opg.optica.org/ao/abstract.cfm?URI=ao-57-18-D155) | [GitHub](https://github.com/hezw2016/DLS-NUC) | Applied Optics (Appl. Opt.) (2018) | 红外图像去噪 / NUC | [BibTeX](#he-dls-nuc-2018) |
| 1D-GF Destriping (Cao16) | Effective Strip Noise Removal for Low-Textured Infrared Images Based on 1-D Guided Filtering | [DOI](https://dl.acm.org/doi/abs/10.1109/TCSVT.2015.2493443) | [GitHub](https://github.com/hezw2016/1D-GF) | IEEE TCSVT (2016) | 红外图像去条带噪声 | [BibTeX](#tcsvt2016-1dgf) |
| DeepIR (Saragadam21) | Thermal Image Processing via Physics-Inspired Deep Networks | [arXiv](https://arxiv.org/abs/2108.07973) | [GitHub](https://github.com/vishwa91/DeepIR) | ICCVW (LCI) (2021) | 红外图像去噪 / 红外图像超分辨率 | [BibTeX](#saragadam2021deepir) |

---

## Visible → Infrared / Thermal Translation

| 简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用 |
|---|---|---|---|---|---|---|
| IRFormer (Chen24) | Implicit Multi-Spectral Transformer: A Lightweight and Effective Visible to Infrared Image Translation Model | [arXiv](https://arxiv.org/abs/2404.07072) | [GitHub](https://github.com/CXH-Research/IRFormer) | IJCNN (2024) | 依据可见光生成红外图像 | [BibTeX](#ijcnn2024-irformer) |
| F-ViTA (Paranjape25) | F-ViTA: Foundation Model Guided Visible to Thermal Translation | [arXiv](https://arxiv.org/abs/2504.02801) | [GitHub](https://github.com/JayParanjape/F-ViTA) | WACV (2026) | 依据可见光生成红外图像 | [BibTeX](#paranjape2025fvita) |
| ThermalGAN (Kniaz18) | ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person Re-Identification in Multispectral Dataset | [PDF](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Kniaz_ThermalGAN_Multimodal_Color-to-Thermal_Image_Translation_for_Person_Re-Identification_in_Multispectral_ECCVW_2018_paper.pdf) | [GitHub](https://github.com/vlkniaz/ThermalGAN) | ECCVW (2018) | 依据可见光生成红外图像 | [BibTeX](#kniaz2018thermalgan) |
| DR-AVIT (Han24) | DR-AVIT: Toward Diverse and Realistic Aerial Visible-to-Infrared Image Translation | [DOI](https://doi.org/10.1109/TGRS.2024.3405989) | [GitHub](https://github.com/silver-hzh/DR-AVIT) | IEEE Transactions on Geoscience and Remote Sensing (TGRS) (2024) | 依据可见光生成红外图像（航拍） | [BibTeX](#tgrs10540003) |
| AVIID Dataset+Baseline (Han23) | Aerial Visible-to-Infrared Image Translation: Dataset, Evaluation, and Baseline | [DOI](https://spj.science.org/doi/10.34133/remotesensing.0096) | [GitHub](https://github.com/silver-hzh/Averial-visible-to-infrared-image-translation) | Journal of Remote Sensing (JRS) (2023) | 可见-红外翻译数据集 + 基线 | [BibTeX](#jrs-han2023aviid) |

---

## Fusion + SR

| 简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用 |
|---|---|---|---|---|---|---|
| DDcGAN | DDcGAN: A Dual-Discriminator Conditional Generative Adversarial Network for Multi-Resolution Image Fusion | [IEEE](https://ieeexplore.ieee.org/document/9031751) | [GitHub](https://github.com/jiayi-ma/DDcGAN) | IEEE Transactions on Image Processing (TIP) (2020) | 多分辨率融合（含红外/可见常用设置） | [BibTeX](#tip2020-ddcgan) |
| HKDnet | Heterogeneous Knowledge Distillation for Simultaneous Infrared-Visible Image Fusion and Super-Resolution | [IEEE](https://ieeexplore.ieee.org/document/9706373) | [GitHub](https://github.com/firewaterfire/HKDnet) | IEEE Transactions on Instrumentation and Measurement (TIM) (2022) | 红外-可见融合 + 超分辨率 | [BibTeX](#tim9706373) |

---

## BibTeX

<a id="teutschcvprw2020"></a>
### TMIQA (Teutsch20)
```bibtex
@InProceedings{TeutschCVPRW2020,
  Title     = {{An Evaluation of Objective Image Quality Assessment for Thermal Infrared Video Tone Mapping}},
  Author    = {Michael Teutsch and Simone Sedelmaier and Sebastian Moosbauer and Gabriel Eilertsen and Thomas Walter},
  Booktitle = {IEEE CVPR Workshops},
  Year      = {2020}
}
