# AGENTS.md — Inscription & Manuscript Digitisation Project
## Writing Reference: Report & Research Paper

> **Purpose of this file:** This is the single source of truth for writing the project
> report and research paper. Any AI agent or human writer must read this fully before
> drafting any section. It contains all technical facts, architectural decisions, scope
> boundaries, results, and team contributions needed to write either document accurately.

---

## WRITING GUIDE — HOW TO USE THIS FILE

### For the Project Report
A project report covers *what was built, how it was built, and how well it works.*
Use this file for: system architecture, pipeline stages, implementation choices, datasets,
evaluation metrics, team roles, timeline, and limitations.
Tone: technical, structured, factual.

### For the Research Paper
A research paper argues *why the approach is novel, what problem it solves, and what
the results demonstrate.* Use this file for: problem statement, motivation, related
work anchors, methodology, experimental results, and future work.
Tone: academic, argumentative, evidence-based.

### Cross-reference table — which sections feed which document

| This file section | Report | Paper |
|---|---|---|
| 1. Project Overview | Introduction | Abstract + Introduction |
| 2. Artefact Types | System Design | Problem Statement |
| 3. Folder Structure | Implementation | Omit (implementation detail) |
| 4. Environment & Dependencies | Appendix | Omit |
| 5. Stage-by-Stage Pipeline | Core Body | Methodology |
| 6. Datasets | Experiments | Datasets section |
| 7. Quality Evaluation | Results & Analysis | Results & Discussion |
| 8. Non-Destructive Rules | System Design | Omit |
| 9. Timeline & Phases | Project Management | Omit |
| 10. Key Decisions & Rationale | Design Choices | Related Work / Justification |
| 11. Limitations & Future Work | Conclusion | Conclusion + Future Work |
| 12. Team Roles | Team / Contributions | Acknowledgements |
| 13. References | References | References |

---

## 1. Project Overview

### Title (use as-is or adapt)
**"Digitisation of Historical Inscriptions and Manuscripts"**

### One-line summary
A software pipeline that takes degraded scanned images of stone inscriptions, palm leaf
manuscripts, copper plates, paper manuscripts, and cave paintings — and produces enhanced,
visually legible images and clean binary outputs ready for downstream OCR in Phase 2.

### Problem statement (for paper introduction)
Millions of historical inscriptions and manuscripts across South Asia remain visually
inaccessible because original artefacts are physically fragile, geographically dispersed,
and degraded by noise, uneven illumination, weathering, and substrate aging. Existing
digital archives (ASI, DLI, eGangotri) provide raw scanned images but no automated
enhancement or binarisation layer that makes the material reliably legible. This project
addresses that gap by building an AI-assisted pipeline that transforms previously
unreadable degraded scans into enhanced, high-contrast, clean binary images ready for
transcription and preservation.

### Goal statement (quote directly in report/paper)
> "Take unclear, degraded images of stone inscriptions, palm leaf manuscripts,
> copper plates, paper manuscripts, and cave/rock paintings — and produce enhanced,
> legible images and clean binary outputs that form the foundation for downstream
> transcription, translation, and archival record assembly."

### Scope — Phase 1 (current, completed)
The project delivers **Stages 1–3** of the pipeline:
1. Preprocessing
2. AI Enhancement
3. Binarisation ← **Phase 1 endpoint**

OCR & Transcription (Stage 4) and all subsequent stages are deferred to **Phase 2**,
per mentor guidance to first deliver a robust, well-evaluated image enhancement and
binarisation system before attempting script recognition.

### Output per processed artefact
- Enhanced image (noise-reduced, super-resolved, colour-corrected)
- Binary image (black text on white background, ready for downstream OCR)

---

## 2. Supported Artefact Types

Five artefact categories are in scope. Each has distinct degradation characteristics
requiring different algorithmic treatment.

| Artefact Type | Primary Degradation | Key Algorithm Used |
|---|---|---|
| Stone inscriptions | Low contrast, surface weathering, shadow | DStretch colour enhancement + Real-ESRGAN |
| Palm leaf manuscripts | Yellowing, ink fading, physical fragility | CLAHE + binarisation + contrast stretch |
| Copper plate inscriptions | Reflective surface, oxidation patina | HDR normalisation + Real-ESRGAN |
| Paper manuscripts | Foxing, stains, torn edges | Denoising + Real-ESRGAN |
| Cave / rock paintings | Uneven lighting, rough irregular texture | DStretch + shadow removal |

**Key point for paper:** The multi-artefact design is a deliberate generalist approach.
Rather than a single-type specialist system, the pipeline routes each image through
artefact-specific processing chains, making it broadly applicable across South Asian
heritage institutions.

---

## 3. System Architecture

### Phase 1 pipeline (Stages 1–3, fully implemented)

```
Raw scanned image input (JPG / TIFF / PNG)
          ↓
Stage 1 — Preprocessing        (normalise, white balance, crop, orient)
          ↓
Stage 2 — AI Enhancement       (denoise, sharpen, super-resolution, DStretch)
          ↓
Stage 3 — Binarisation         (separate text pixels from background)
          ↓
        [Binary PNG output — Phase 1 endpoint]
          ↓
Stage 4 — OCR / Transcription  (Phase 2 — future)
          ↓
Stage 5 — Translation          (Phase 3 — future)
          ↓
Stage 6+ — Record Assembly, Storage, Web UI  (future phases)
```

### Technology stack (Phase 1 only)

| Layer | Technology | Version |
|---|---|---|
| Image processing core | OpenCV | 4.9.0.80 |
| Image manipulation | Pillow | 10.3.0 |
| Scientific image ops | scikit-image | 0.23.2 |
| Numerical computing | NumPy | 1.26.4 |
| AI super-resolution | Real-ESRGAN (basicsr) | 0.3.0 / 1.4.2 |
| Deep learning runtime | PyTorch | ≥ 2.0.0 |

### Model weights used

| Model | Purpose | Source |
|---|---|---|
| `RealESRGAN_x4plus.pth` | General super-resolution for inscriptions | Wang et al. 2021 |
| `RealESRGAN_x4plus_anime_6B.pth` | Line-art style carvings | Wang et al. 2021 |

---

## 4. Stage-by-Stage Technical Detail

### Stage 1 — Preprocessing (`src/preprocess.py`) ✅ Implemented

**Purpose:** Normalise raw images to a consistent baseline before AI processing.

**Functions implemented:**
- `load_image(path)` — Loads image as BGR numpy array, correcting EXIF orientation
  via `PIL.ImageOps.exif_transpose` so output is visually upright regardless of camera
  orientation. EXIF tag is baked into pixels, not carried in metadata.
- `normalise_brightness(img)` — CLAHE histogram equalisation
  (`clipLimit=2.0, tileGridSize=(8,8)`) for uneven lighting common in field photography.
- `auto_white_balance(img)` — Grey-world assumption colour correction.
- `crop_borders(img, threshold=10)` — Removes blank/dark scanner margins.
- `preprocess(img_path, output_path)` — Full chain; saves as JPEG quality=95.
- `process_directory(input_dir, output_dir)` — Batch preprocessing.

**Output format:** JPEG (quality=95). TIFF was considered but rejected — PIL/libtiff
metadata write failures on Windows.

---

### Stage 2 — AI Enhancement (`src/enhance.py`) ✅ Implemented

**Purpose:** Core value-add of the project. Makes previously unreadable inscriptions
legible using AI super-resolution and colour science.

**Functions:**
- `denoise(img, strength=10)` — Non-local means denoising.
- `enhance_with_realesrgan(img, scale=2, model_path=...)` — Super-resolution.
  Uses `outscale=2` (from a 4× model) to avoid over-smoothing.
- `dstretch(img, colour_space="LAB")` — DStretch decorrelation stretch. Reveals
  colour differences invisible to the human eye via eigendecomposition of colour
  channel covariance. Colour space options: LAB, YDS, YBK, LDS.
- `sharpen(img, amount=1.5)` — Unsharp mask sharpening.
- `enhance(img_path, output_path, use_dstretch=False)` — Full chain.

**Per-artefact processing chains:**
- Stone inscriptions: denoise → Real-ESRGAN → sharpen
- Cave paintings: denoise → DStretch (YBK) → sharpen
- Palm leaf manuscripts: auto_white_balance → denoise → Real-ESRGAN
- Copper plates: HDR normalisation → denoise → Real-ESRGAN
- Paper manuscripts: denoise → Real-ESRGAN → sharpen

**Real-ESRGAN configuration:**
```
RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
tile=400, tile_pad=10, half=False, outscale=2
```

---

### Stage 3 — Binarisation (`src/binarise.py`) ✅ Implemented

**Purpose:** Convert enhanced colour image to clean binary (black text, white background).

**Functions:**
- `binarise_sauvola(img, window_size=25)` — Sauvola local adaptive thresholding.
  **Preferred method** for most inscription types. Handles spatially uneven backgrounds.
- `binarise_otsu(img)` — Otsu global thresholding. Suitable for clean paper manuscripts.
- `binarise_adaptive(img)` — OpenCV adaptive mean thresholding. Fallback for mixed quality.
- `remove_noise_blobs(binary, min_size=50)` — Removes disconnected components
  smaller than 50 pixels (dust, surface damage artefacts).
- `binarise(img_path, output_path, method="sauvola")` — Main entry point.

**Post-binarisation step:** Morphological closing (`cv2.MORPH_CLOSE`, 2×2 kernel)
reconnects broken character strokes — important for weathered inscriptions.

**Output format:** PNG (lossless — no compression artefacts on binary data).

---

## 5. Interdisciplinary Team Roles

### CS / IT — AI Pipeline & Software

| Responsibility | File |
|---|---|
| Preprocessing | `src/preprocess.py` |
| AI enhancement (Real-ESRGAN, DStretch) | `src/enhance.py` |
| Binarisation | `src/binarise.py` |

---

### ECE — Signal Processing & Image Quality Analysis

**Contribution 1 — Noise characterisation & modelling**
Profiles noise types present in scanned inscription images across all five artefact
categories: Gaussian, salt-and-pepper, JPEG compression artefacts, and periodic noise.
Deliverable: `docs/noise_analysis_report.pdf`

**Contribution 2 — Custom filter design (`src/filters.py`)**
- `gabor_filter_bank(img, frequencies=[0.1,0.2,0.4], orientations=8)` — 24-filter
  bank for inscription texture separation.
- `directional_edge_enhance(img, angle_deg=45.0)` — Orientation-selective sharpening.
- `remove_periodic_noise_fft(img, threshold=0.1)` — FFT-based scanner line removal.

**Contribution 3 — Histogram & colour channel analysis (`src/analysis.py`)**
- Per-channel statistics (mean, std, skewness, kurtosis).
- DStretch colour space tuning (LAB vs YDS vs YBK vs LDS) per material.
- Before/after histogram plots.

**Contribution 4 — Image quality metrics (`src/metrics.py`)**
- `compute_psnr(original, enhanced)` — Target ≥ 30 dB.
- `compute_ssim(original, enhanced)` — Target ≥ 0.85.
- `compute_cnr(img, text_mask)` — Text-to-background contrast ratio.
- `compute_sharpness(img)` — Laplacian variance as edge sharpness proxy.
- `full_quality_report(original, enhanced, text_mask)` — Consolidated quality dict.

---

### IEM — Process Design, Project Management & Impact Analysis

**Contribution 1 — Project management**
Gantt chart, sprint planning, risk register.
Deliverable: `docs/project_plan.xlsx`

**Contribution 2 — Digitisation workflow design**
Value stream mapping identifying Real-ESRGAN (Stage 2, 2 s/image) as the pipeline
bottleneck. Total pipeline: ~2.8 s/image (~21 images/minute on a single GPU).
Deliverable: `docs/workflow_analysis.pdf`

**Contribution 3 — Cost-benefit & scalability analysis**
- Compute cost per image enhancement (GPU cloud vs local)
- Storage cost per artefact (enhanced JPEG + binary PNG)
- Researcher time saved vs manual image preparation
- Scalability model: 100 → 10,000 → 1,000,000 images
Deliverable: `docs/cost_benefit_analysis.pdf`

**Contribution 4 — Stakeholder documentation**
Non-technical overview, data governance policy, impact statement for ASI / museum curators.
Deliverables: `docs/user_guide.pdf`, `docs/impact_statement.pdf`

---

### Ownership summary table

| Component | Owner |
|---|---|
| Preprocessing pipeline | CS/IT |
| AI enhancement (Real-ESRGAN, DStretch) | CS/IT |
| Binarisation | CS/IT |
| Noise modelling & characterisation | ECE |
| Custom filter design (Gabor, FFT) | ECE |
| Colour channel & histogram analysis | ECE |
| Image quality metrics (PSNR, SSIM, CNR) | ECE |
| Project schedule & risk register | IEM |
| Value stream mapping & throughput | IEM |
| Cost-benefit & scalability analysis | IEM |
| Stakeholder & impact documentation | IEM |

---

## 6. Datasets

### Primary datasets (used for development and evaluation)

| Dataset | Contents | Access |
|---|---|---|
| Ancient Tamil Stone Inscriptions | Tamil stone inscriptions with LiDAR and 3D models | kaggle.com/datasets/athiraishanmugam/ancient-tamil-stone-inscriptions |
| Tamil Handwritten Palm Leaf (THPLMD) | 262 deteriorated Tamil palm leaf samples with binarised ground truth | ScienceDirect / PMC |
| Tamil Handwritten Character Corpus | Tamil handwritten characters across centuries | data.mendeley.com/datasets/6zcpgchvmx/1 |

### Secondary datasets

| Dataset | Contents | Access |
|---|---|---|
| Indiscapes (IIIT Hyderabad) | Layout annotations for historical Indic manuscripts | ihdia.iiit.ac.in |
| Brahmi Character Dataset | 1032 Ashokan Brahmi characters, 258 classes (Phase 2 OCR training) | arxiv.org/abs/2501.01981 |
| Kannada Inscriptions | Leaf manuscripts and stone inscriptions from Hampi | Kuvempu Institute of Kannada Studies |

### Institutional archives

| Source | URL |
|---|---|
| Digital Library of India | dli.ernet.in |
| eGangotri | egangotri.org |
| IIIT Hyderabad IHDIA | ihdia.iiit.ac.in |
| Archaeological Survey of India | asi.nic.in |

---

## 7. Quality Evaluation & Results

### Metrics tracked per processed image

| Metric | What it measures | Tool | Target |
|---|---|---|---|
| PSNR | Signal-to-noise ratio after enhancement | `skimage.metrics.peak_signal_noise_ratio` | ≥ 30 dB |
| SSIM | Structural similarity to original | `skimage.metrics.structural_similarity` | ≥ 0.85 |
| CNR | Contrast-to-noise ratio (text vs background) | `src/metrics.py` (ECE) | Higher = better |
| Edge sharpness | Laplacian variance | `src/metrics.py` (ECE) | Higher = better |

### Phase 1 results summary (38-image test set)

| Metric | Overall result | Target met |
|---|---|---|
| Mean PSNR | 32.3 dB | ✅ (≥ 30 dB) |
| Mean SSIM | 0.88 | ✅ (≥ 0.85) |
| CNR improvement | 1.4–3.1 → 3.2–5.8 (2–2.5× gain) | ✅ |
| Sharpness improvement | 200–420 → 510–920 (~2× gain) | ✅ |
| Throughput | ~21 images/minute (2.8 s/image total) | ✅ |

---

## 8. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Real-ESRGAN over bicubic upscaling | Trained on real-world degraded photos; preserves texture and character stroke edges far better than classical upscaling |
| DStretch for rock / cave art | Originally developed for rock art analysis by Jon Harman; reveals colour differences invisible to the human eye through decorrelation stretch |
| Sauvola over Otsu binarisation | Handles spatially uneven backgrounds (stone surface, aged palm leaf fibre) better than global threshold methods |
| JPEG (quality=95) for preprocessing output | Avoids PIL/libtiff metadata write failures on Windows; quality=95 is indistinguishable from lossless for all subsequent pipeline stages |
| `outscale=2` for Real-ESRGAN | Uses 4× model but outputs at 2× resolution — avoids hallucinated detail on dense ancient script |
| Phase 1 endpoint at binarisation | Mentor guidance: deliver robust, well-evaluated image enhancement before OCR; ensures binarisation quality is validated independently before script recognition is added |

---

## 9. Implementation Phases & Timeline

| Phase | Deliverable | Duration | Status |
|---|---|---|---|
| 1 | Environment setup, folder structure, 5 sample images tested | Week 1 | ✅ Complete |
| 2 | `preprocess.py` + `enhance.py` — full enhancement pipeline with batch processing | Weeks 2–3 | ✅ Complete |
| 3 | `binarise.py` — binarisation for all five artefact types | Weeks 4–6 | ✅ Complete |
| 4 (future) | `ocr.py` — OCR & transcription (Tesseract + EasyOCR ensemble) | Phase 2 | 🔲 Future |
| 5 (future) | Translation, record assembly, web UI | Phase 3+ | 🔲 Future |

---

## 10. Non-Destructive Processing Policy

1. **Never overwrite originals.** `data/raw/` is read-only. All outputs go to
   `data/enhanced/`, `data/binarised/`.
2. **3-2-1 backup rule:** 3 copies of original data, on 2 different storage media,
   with 1 offsite/cloud backup.
3. **Preserve EXIF metadata** through all processing stages.
4. **Log every operation** — stage, timestamp, pipeline version, model weights version.
5. **Version control model weights** — every run logs the exact Real-ESRGAN weights
   file used, ensuring reproducibility of any processed image.

---

## 11. Limitations & Future Work

### Current limitations (Phase 1)

- **OCR not yet implemented.** The pipeline endpoint is a clean binary image. Stage 4
  (OCR) using a Tesseract + EasyOCR ensemble for Tamil, Sanskrit, Kannada, Telugu,
  Malayalam, and Devanagari is the primary Phase 2 deliverable.

- **Heavy damage threshold.** Images with >50% surface damage produce degraded binary
  output. Planned fix: LaMa inpainting before binarisation.

- **3D inscriptions.** Deep-carved stone inscriptions benefit from 3D/LiDAR data.
  The Kaggle dataset includes 3D models; processing 3D point clouds is Phase 3 work.

- **Multi-script inscriptions.** Some artefacts combine two scripts. Future work:
  segment by script region before OCR routing.

### Future work

- **Phase 2 — OCR & Transcription:** Tesseract + EasyOCR ensemble for six Indic scripts;
  custom Brahmi OCR model trained on 1032-character Ashokan Brahmi dataset.
- **Phase 3 — Translation:** Helsinki-NLP OPUS-MT for post-10th century CE texts;
  LLM fallback for archaic classical forms.
- **Phase 4 — Record Assembly & Export:** JSON records, PDF export, structured metadata.
- **Phase 5 — Web Interface:** React + FastAPI portal for institutional use.
- LaMa inpainting for heavily damaged images.
- 3D point cloud processing for deep-carved stone inscriptions.
- Public-facing open-access research portal (long-term).

---

## 12. References

- **Real-ESRGAN:** Wang, X. et al. (2021). Real-ESRGAN: Training Real-World Blind
  Super-Resolution with Pure Synthetic Data. arXiv:2107.10833
- **DStretch:** Harman, J. (2008). Using Decorrelation Stretch to Enhance Rock Art Images.
- **SSIM metric:** Wang, Z. et al. (2004). Image quality assessment: from error
  visibility to structural similarity. *IEEE Trans. Image Processing*, 13(4), 600–612.
- **Gabor filters:** Daugman, J. G. (1985). Uncertainty relation for resolution in
  space, spatial frequency, and orientation. *JOSA A*, 2(7), 1160–1169.
- **THPLMD / Sauvola:** Dhanya et al. (2020). Tamil Palm Leaf Manuscript Binarisation.
- **Brahmi OCR dataset:** arXiv:2501.01981
- **Indiscapes dataset:** IIIT Hyderabad — ihdia.iiit.ac.in

---

## 13. Project Folder Structure (for report appendix)

```
inscription-digitisation/
├── AGENTS.md                    ← this file
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── raw/                     ← original scans (read-only)
│   ├── enhanced/                ← Stage 2 output
│   └── binarised/               ← Stage 3 output
├── models/
│   └── weights/
│       ├── RealESRGAN_x4plus.pth
│       └── RealESRGAN_x4plus_anime_6B.pth
├── src/
│   ├── preprocess.py            ← Stage 1
│   ├── enhance.py               ← Stage 2
│   ├── binarise.py              ← Stage 3
│   ├── filters.py               ← ECE: Gabor, FFT, directional filters
│   ├── analysis.py              ← ECE: colour distribution tools
│   ├── metrics.py               ← ECE: PSNR, SSIM, CNR, sharpness
│   └── utils.py                 ← shared helpers
├── docs/
│   ├── noise_analysis_report.pdf     ← ECE deliverable
│   ├── project_plan.xlsx             ← IEM deliverable
│   ├── workflow_analysis.pdf         ← IEM deliverable
│   ├── cost_benefit_analysis.pdf     ← IEM deliverable
│   ├── user_guide.pdf                ← IEM deliverable
│   └── impact_statement.pdf         ← IEM deliverable
├── tests/
│   ├── test_preprocess.py       ← 4 unit tests
│   └── test_enhance.py          ← 1 integration test
└── outputs/
    └── logs/                    ← processing logs
```

---

*Last updated: June 2026.*
*Phase 1 (Stages 1–3: Preprocessing, Enhancement, Binarisation) is complete.*
*Phase 2 (OCR & Transcription) is the next planned deliverable.*
*Agent or writer: read this file top-to-bottom before drafting any section.*
