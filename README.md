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
| CoRPLE | Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00391.pdf) | [GitHub](https://github.com/hey-it-s-me/CoRPLE) | ECCV 2024 | 红外图像超分辨率 | [BibTeX](#bibtex-corple) |

---

## Infrared Image Enhancement

| 简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用 |
|---|---|---|---|---|---|---|
| IE-CGAN (Kuang19) | Single infrared image enhancement using a deep convolutional neural network | [DOI](https://dl.acm.org/doi/10.1016/j.neucom.2018.11.081) | [GitHub](https://github.com/Kuangxd/IE-CGAN) | Neurocomputing (2019) | 红外图像对比度增强 | [BibTeX](#neucom2019-iecgan) |
| TEN (Choi16) | Thermal Image Enhancement using Convolutional Neural Network | [IEEE](https://ieeexplore.ieee.org/document/7759059) | [GitHub](https://github.com/ninadakolekar/Thermal-Image-Enhancement?tab=readme-ov-file) | IROS (2016) | 热红外图像增强（CNN） | [BibTeX](#iros7759059) |

---

## Infrared Denoising / NUC / Destriping

## Infrared Denoising / NUC / Destriping

| Method | Title | Paper Link | Code Link | Journal / Conference | Task | Method |
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





## Visible → Infrared / Thermal Translation

| 简称 | 标题 | 论文路径 | 代码路径 | 发表期刊/会议 | 对应任务 | 引用 |
|---|---|---|---|---|---|---|
| IRFormer (Chen24) | Implicit Multi-Spectral Transformer: A Lightweight and Effective Visible to Infrared Image Translation Model | [arXiv](https://arxiv.org/abs/2404.07072) | [GitHub](https://github.com/CXH-Research/IRFormer) | IJCNN (2024) | 依据可见光生成红外图像 | [BibTeX](#ijcnn2024-irformer) |
| F-ViTA (Paranjape25) | F-ViTA: Foundation Model Guided Visible to Thermal Translation | [arXiv](https://arxiv.org/abs/2504.02801) | [GitHub](https://github.com/JayParanjape/F-ViTA) | WACV (2026) | 依据可见光生成红外图像 | [BibTeX](#paranjape2025fvita) |
| ThermalGAN (Kniaz18) | ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person Re-Identification in Multispectral Dataset | [PDF](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Kniaz_ThermalGAN_Multimodal_Color-to-Thermal_Image_Translation_for_Person_Re-Identification_in_Multispectral_ECCVW_2018_paper.pdf) | [GitHub](https://github.com/vlkniaz/ThermalGAN) | ECCVW (2018) | 依据可见光生成红外图像 | [BibTeX](#kniaz2018thermalgan) |
| DR-AVIT (Han24) | DR-AVIT: Toward Diverse and Realistic Aerial Visible-to-Infrared Image Translation | [DOI](https://doi.org/10.1109/TGRS.2024.3405989) | [GitHub](https://github.com/silver-hzh/DR-AVIT) | IEEE Transactions on Geoscience and Remote Sensing (TGRS) (2024) | 依据可见光生成红外图像（航拍） | [BibTeX](#tgrs10540003) |
| AVIID Dataset+Baseline (Han23) | Aerial Visible-to-Infrared Image Translation: Dataset, Evaluation, and Baseline | [DOI](https://spj.science.org/doi/10.34133/remotesensing.0096) | [GitHub](https://github.com/silver-hzh/Averial-visible-to-infrared-image-translation) | Journal of Remote Sensing (JRS) (2023) | 可见-红外翻译数据集 + 基线 | [BibTeX](#jrs-han2023aviid) |
| InfraGAN | InfraGAN: A GAN architecture to transfer visible images to infrared domain | [Paper](https://www.sciencedirect.com/science/article/pii/S0167865522000332?) | [GitHub](https://github.com/makifozkanoglu/InfraGAN) | Pattern Recognition Letters 2022 | 可见光 → 红外域 / 域迁移 / GAN | [BibTeX](#bibtex-infragan) |
| ClawGAN | ClawGAN: Claw connection-based generative adversarial networks for facial image translation in thermal to RGB visible light | [Paper](https://www.sciencedirect.com/science/article/pii/S0957417421015785?) | [GitHub](https://github.com/Luoyi3819/ClawGAN) | Expert Systems with Applications 2022 | 热红外（thermal）→ 可见光RGB / 人脸翻译 | [BibTeX](#bibtex-clawgan) |
| EdgeGuided-RGB2TIR | Edge-guided Multi-domain RGB-to-TIR image Translation for Training Vision Tasks with Challenging Labels | [arXiv](https://arxiv.org/abs/2301.12689) | [GitHub](https://github.com/RPM-Robotics-Lab/sRGB-TIR) | ICRA 2023 | RGB → TIR 翻译；用于生成TIR训练数据（光流/检测等） | [BibTeX](#bibtex-edgeguided-rgb2tir) |
| PID | PID: Physics-Informed Diffusion Model for Infrared Image Generation | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320325004765) | [GitHub](https://github.com/fangyuanmao/PID) | Pattern Recognition (Vol.169) 2026 | RGB → IR 生成 / 扩散模型 + 物理约束 | [BibTeX](#bibtex-pid) |

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
```

<a id="dglee-2024-tcnet"></a>

### TCNet (ThermalChameleon)

```bibtex
@ARTICLE {dglee-2024-tcnet,
    AUTHOR = { Dong-Guw Lee and Jeongyun Kim and Younggun Cho and Ayoung Kim },
    TITLE = { Thermal Chameleon: Task-Adaptive Tone-mapping for Radiometric Thermal-Infrared images },
    JOURNAL = {IEEE Robotics and Automation Letters (RA-L) },
    YEAR = { 2024 }
}
```

<a id="wacvw9707563"></a>

### TFDL

```bibtex
@INPROCEEDINGS{9707563,
 author = {Hu, Litao and Chen, Huaijin and Allebach, Jan P.},
 title = {Joint Multi-Scale Tone Mapping and Denoising for HDR Image Enhancement},
 booktitle = {2022 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
 year = {2022},
 pages = {729-738},
 doi = {10.1109/WACVW54805.2022.00080}
}
```

<a id="li2025difiisr"></a>

### DifIISR (Li25)

```bibtex
@inproceedings{li2025difiisr,
  title={Difiisr: A diffusion model with gradient guidance for infrared image super-resolution},
  author={Li, Xingyuan and Wang, Zirui and Zou, Yang and Chen, Zhixin and Ma, Jun and Jiang, Zhiying and Ma, Long and Liu, Jinyuan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={7534--7544},
  year={2025}
}
```

<a id="gupta2021toward"></a>

### UGSR (Gupta22)

```bibtex
@article{gupta2021toward,
  title={Toward Unaligned Guided Thermal Super-Resolution},
  author={Gupta, Honey and Mitra, Kaushik},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={433--445},
  year={2021},
  publisher={IEEE}
}
```

<a id="cvprw9150703"></a>

### TherISuRNet (Chudasama20)

```bibtex
@INPROCEEDINGS{9150703,
  author={Chudasama, Vishal and Patel, Heena and Prajapati, Kalpesh and Upla, Kishor and Ramachandra, Raghavendra and Raja, Kiran and Busch, Christoph},
  booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
  title={TherISuRNet - A Computationally Efficient Thermal Image Super-Resolution Network}, 
  year={2020},
  pages={388-397},
  doi={10.1109/CVPRW50498.2020.00051}
}
```

<a id="tcsvt8432397"></a>

### CDN_MRF (He19)

```bibtex
@ARTICLE{8432397,
  author={He, Zewei and Tang, Siliang and Yang, Jiangxin and Cao, Yanlong and Ying Yang, Michael and Cao, Yanpeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Cascaded Deep Networks With Multiple Receptive Fields for Infrared Image Super-Resolution}, 
  year={2019},
  volume={29},
  number={8},
  pages={2310-2322},
  doi={10.1109/TCSVT.2018.2864777}
}
```

<a id="spl9424970"></a>

### PSRGAN / PSRGAN-TL (Huang21)

```bibtex
@ARTICLE{9424970, 
  author={Huang, Yongsong and Jiang, Zetao and Lan, Rushi and Zhang, Shaoqin and Pi, Kui}, 
  journal={IEEE Signal Processing Letters}, 
  title={Infrared Image Super-Resolution via Transfer Learning and PSRGAN}, 
  year={2021}, 
  volume={28}, 
  pages={982-986}, 
  doi={10.1109/LSP.2021.3077801}
}
```

<a id="cvprw10678449"></a>

### TISR-Diffusion (Cortes24)

```bibtex
@INPROCEEDINGS{10678449,
  author={Cortés-Mendez, Carlos and Hayet, Jean-Bernard},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
  title={Exploring the usage of diffusion models for thermal image super-resolution: a generic, uncertainty-aware approach for guided and non-guided schemes}, 
  year={2024},
  pages={3123-3130},
  doi={10.1109/CVPRW63382.2024.00318}
}
```

<a id="tip2020-ddcgan"></a>

### DDcGAN

```bibtex
@article{ma2020ddcgan,
  title={DDcGAN: A Dual-discriminator Conditional Generative Adversarial Network for Multi-resolution Image Fusion},
  author={Ma, Jiayi and Xu, Han and Jiang, Junjun and Mei, Xiaoguang and Zhang, Xiao-Ping},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4980--4995},
  year={2020},
  publisher={IEEE}
}
```

<a id="tim9706373"></a>

### HKDnet

```bibtex
@ARTICLE{9706373,
  author={Xiao, Wanxin and Zhang, Yafei and Wang, Hongbin and Li, Fan and Jin, Hua},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Heterogeneous Knowledge Distillation for Simultaneous Infrared-Visible Image Fusion and Super-Resolution}, 
  year={2022},
  volume={71},
  pages={1-15},
  doi={10.1109/TIM.2022.3149101}
}
```

<a id="neucom2019-iecgan"></a>

### IE-CGAN (Kuang19)

```bibtex
@article{10.1016/j.neucom.2018.11.081,
  author = {Kuang, Xiaodong and Sui, Xiubao and Liu, Yuan and Chen, Qian and Gu, Guohua},
  title = {Single infrared image enhancement using a deep convolutional neural network},
  year = {2019},
  journal = {Neurocomputing},
  volume = {332},
  number = {C},
  pages = {119--128},
  doi = {10.1016/j.neucom.2018.11.081}
}
```

<a id="shen2025310"></a>

### IDTransformer (Shen25)

```bibtex
@article{SHEN2025310,
  title = {IDTransformer: Infrared image denoising method based on convolutional transposed self-attention},
  journal = {Alexandria Engineering Journal},
  volume = {110},
  pages = {310-321},
  year = {2025},
  issn = {1110-0168},
  doi = {https://doi.org/10.1016/j.aej.2024.09.101},
  url = {https://www.sciencedirect.com/science/article/pii/S1110016824011256},
  author = {Zhengwei Shen and Feiwei Qin and Ruiquan Ge and Changmiao Wang and Kai Zhang and Jie Huang},
  keywords = {Image denoising, Infrared image, Self-attention, Feature fusion}
}
```

<a id="barral_2024_wacv"></a>

### MultiView-FPN (Barral24)

```bibtex
@InProceedings{Barral_2024_WACV,
    author    = {Barral, Hortensia and Arias, Pablo and Davy, Axel},
    title     = {Fixed Pattern Noise Removal for Multi-View Single-Sensor Infrared Camera},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {1669-1678}
}
```

<a id="he-dls-nuc-2018"></a>

### DLS-NUC (He18)

```bibtex
@article{He-DLS-NUC-2018,
author = {Zewei He and Yanpeng Cao and Yafei Dong and Jiangxin Yang and Yanlong Cao and Christel-L""{o}ic Tisse},
journal = {Appl. Opt.},
number = {18},
pages = {D155--D164},
title = {Single-image-based nonuniformity correction of uncooled long-wave infrared detectors: a deep-learning approach},
volume = {57},
month = {Jun},
year = {2018},
doi = {10.1364/AO.57.00D155}
}
```

<a id="tcsvt2016-1dgf"></a>

### 1D-GF Destriping (Cao16)

```bibtex
@article{10.1109/TCSVT.2015.2493443,
author = {Cao, Yanpeng and Yang, Michael Ying and Tisse, Christel-Loic},
title = {Effective Strip Noise Removal for Low-Textured Infrared Images Based on 1-D Guided Filtering},
year = {2016},
issue_date = {December 2016},
publisher = {IEEE Press},
volume = {26},
number = {12},
issn = {1051-8215},
url = {https://doi.org/10.1109/TCSVT.2015.2493443},
doi = {10.1109/TCSVT.2015.2493443},
journal = {IEEE Trans. Cir. and Sys. for Video Technol.},
month = dec,
pages = {2176–2188},
numpages = {13}
}
```

<a id="saragadam2021deepir"></a>

### DeepIR (Saragadam21)

```bibtex
@misc{saragadam2021deepir,
  title={Thermal Image Processing via Physics-Inspired Deep Networks},
  author={Vishwanath Saragadam and Akshat Dave and Ashok Veeraraghavan and Richard G. Baraniuk},
  year={2021},
  note={IEEE Intl. Conf. Computer Vision Workshop on Learning for Computational Imaging (ICCVW-LCI)}
}
```

<a id="ijcnn2024-irformer"></a>

### IRFormer (Chen24)

```bibtex
@inproceedings{DBLP:conf/ijcnn/ChenCZLZL24,
  author       = {Yijia Chen and
                  Pinghua Chen and
                  Xiangxin Zhou and
                  Yingtie Lei and
                  Ziyang Zhou and
                  Mingxian Li},
  title        = {Implicit Multi-Spectral Transformer: An Lightweight and Effective
                  Visible to Infrared Image Translation Model},
  booktitle    = {International Joint Conference on Neural Networks},
  pages        = {1--8},
  year         = {2024}
}
```

<a id="paranjape2025fvita"></a>

### F-ViTA (Paranjape25)

```bibtex
@misc{paranjape2025fvitafoundationmodelguided,
      title={F-ViTA: Foundation Model Guided Visible to Thermal Translation}, 
      author={Jay N. Paranjape and Celso de Melo and Vishal M. Patel},
      year={2025},
      eprint={2504.02801},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.02801}
}
```

<a id="kniaz2018thermalgan"></a>

### ThermalGAN (Kniaz18)

```bibtex
@InProceedings{Kniaz2018,
author={Kniaz, Vladimir V. and
Knyaz, Vladimir A. and
Hlad\r{u}vka, Ji{\v r}{\'{\i}}  and Kropatsch, Walter G. and Mizginov, Vladimir A.},
title={{ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person Re-Identification in Multispectral Dataset}},
booktitle={{Computer Vision -- ECCV 2018 Workshops}},
year={2018},
publisher={Springer International Publishing}
}
```

<a id="tgrs10540003"></a>

### DR-AVIT (Han24)

```bibtex
@ARTICLE{10540003,
  author={Han, Zonghao and Zhang, Shun and Su, Yuru and Chen, Xiaoning and Mei, Shaohui},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DR-AVIT: Toward Diverse and Realistic Aerial Visible-to-Infrared Image Translation}, 
  year={2024},
  volume={62},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2024.3405989}
}
```

<a id="jrs-han2023aviid"></a>

### AVIID Dataset+Baseline (Han23)

```bibtex
@article{han2023aviid,
  title={Aerial Visible-to-Infrared Image Translation: Dataset, Evaluation, and Baseline},
  author={Han, Zonghao and others},
  journal={Journal of Remote Sensing},
  year={2023},
  doi={10.34133/remotesensing.0096}
}
```

<a id="iros7759059"></a>

### TEN (Choi16)

```bibtex
@INPROCEEDINGS{7759059,
  author={Choi, Yukyung and Kim, Namil and Hwang, Soonmin and Kweon, In So},
  booktitle={2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Thermal Image Enhancement using Convolutional Neural Network}, 
  year={2016},
  pages={223-230},
  doi={10.1109/IROS.2016.7759059}
}
```

<a id="tip2020-ddcgan"></a>

### DDcGAN

```bibtex
@article{ma2020ddcgan,
  title={DDcGAN: A Dual-discriminator Conditional Generative Adversarial Network for Multi-resolution Image Fusion},
  author={Ma, Jiayi and Xu, Han and Jiang, Junjun and Mei, Xiaoguang and Zhang, Xiao-Ping},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4980--4995},
  year={2020},
  publisher={IEEE}
}
```

<a id="tim9706373"></a>

### HKDnet

```bibtex
@ARTICLE{9706373,
  author={Xiao, Wanxin and Zhang, Yafei and Wang, Hongbin and Li, Fan and Jin, Hua},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Heterogeneous Knowledge Distillation for Simultaneous Infrared-Visible Image Fusion and Super-Resolution}, 
  year={2022},
  volume={71},
  pages={1-15},
  doi={10.1109/TIM.2022.3149101}
}
```


### CoRPLE

<a id="bibtex-corple"></a>
```bibtex
@inproceedings{li2024contourlet,
  title={Contourlet residual for prompt learning enhanced infrared image super-resolution},
  author={Li, Xingyuan and Liu, Jinyuan and Chen, Zhixin and Zou, Yang and Ma, Long and Fan, Xin and Liu, Risheng},
  booktitle={European Conference on Computer Vision},
  pages={270--288},
  year={2024},
  organization={Springer}
}
```

### SNRWDNN
<a id="bibtex-snrwdnn"></a>
```bibtex
@ARTICLE{8678750,
  author={Guan, Juntao and Lai, Rui and Xiong, Ai},
  journal={IEEE Access},
  title={Wavelet Deep Neural Network for Stripe Noise Removal},
  year={2019},
  volume={7},
  number={},
  pages={44544-44554},
  keywords={Wavelet domain;Wavelet transforms;Training;Neural networks;Deep learning;Image denoising;Neural networks;image denoising;infrared image sensors;wavelet transforms},
  doi={10.1109/ACCESS.2019.2908720}
}
```

### InfraGAN
<a id="bibtex-infragan"></a>
```bibtex
@article{OZKANOGLU202269,
  title = {InfraGAN: A GAN architecture to transfer visible images to infrared domain},
  journal = {Pattern Recognition Letters},
  volume = {155},
  pages = {69-76},
  year = {2022},
  issn = {0167-8655},
  doi = {https://doi.org/10.1016/j.patrec.2022.01.026},
  url = {https://www.sciencedirect.com/science/article/pii/S0167865522000332},
  author = {Mehmet Akif Özkanoğlu and Sedat Ozer},
  keywords = {Domain transfer, GANs, Infrared image generation},
  abstract = {Utilizing both visible and infrared (IR) images in various deep learning based computer vision tasks has been a recent trend. Consequently, datasets having both visible and IR image pairs are desired in many applications. However, while large image datasets taken at the visible spectrum can be found in many domains, large IR-based datasets are not easily available in many domains. The lack of IR counterparts of the available visible image datasets limits existing deep algorithms to perform on IR images effectively. In this paper, to overcome with that challenge, we introduce a generative adversarial network (GAN) based solution and generate the IR equivalent of a given visible image by training our deep network to learn the relation between visible and IR modalities. In our proposed GAN architecture (InfraGAN), we introduce using structural similarity as an additional loss function. Furthermore, in our discriminator, we do not only consider the entire image being fake or real but also each pixel being fake or real. We evaluate our comparative results on three different datasets and report the state of the art results over five metrics when compared to Pix2Pix and ThermalGAN architectures from the literature. We report up to +16% better performance in Structural Similarity Index Measure (SSIM) over Pix2Pix and +8% better performance over ThermalGAN for VEDAI dataset. Further gains on different metrics and on different datasets are also reported in our experiments section.}
}
```

### ClawGAN
<a id="bibtex-clawgan"></a>
```bibtex
@article{Luo2022ClawGAN,
  author = {Yi Luo and Dechang Pi and Yue Pan and Lingqiang Xie and Wen Yu and Yufei Liu},
  title = {ClawGAN: Claw connection-based generative adversarial networks for facial image translation in thermal to {RGB} visible light},
  journal = {Expert Systems with Applications},
  year = {2022},
  volume = {191},
  pages = {116269}
}
```

### EdgeGuided-RGB2TIR
<a id="bibtex-edgeguided-rgb2tir"></a>
```bibtex
@ARTICLE{lee-2023-edgemultiRGB2TIR,
  author={Lee, Dong-Guw and Kim, Ayoung},
  conference={IEEE International Conference on Robotics and Automation},
  title={Edge-guided Multi-domain RGB-to-TIR image Translation for Training Vision Tasks with Challenging Labels},
  year={2023},
  status={underreview}
}
```

### PID
<a id="bibtex-pid"></a>
```bibtex
@article{mao2026pid,
  title={PID: physics-informed diffusion model for infrared image generation},
  author={Mao, Fangyuan and Mei, Jilin and Lu, Shun and Liu, Fuyang and Chen, Liang and Zhao, Fangzhou and Hu, Yu},
  journal={Pattern Recognition},
  volume={169},
  pages={111816},
  year={2026},
  publisher={Elsevier}
}
```
