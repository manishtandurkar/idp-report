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
**"Digitisation of Historical
Inscriptions and Manuscripts"**

### One-line summary
An end-to-end software pipeline that takes degraded scanned images of stone
inscriptions, palm leaf manuscripts, copper plates, paper manuscripts, and cave
paintings — and produces clean, searchable, citable digital records containing an
enhanced image, a Unicode transcription, and structured metadata.

### Problem statement (for paper introduction)
Millions of historical inscriptions and manuscripts across South Asia remain inaccessible
to researchers and the public because original artefacts are physically fragile,
geographically dispersed, and visually degraded beyond easy reading. Manual
transcription is slow, expensive, and requires rare specialist expertise in ancient
scripts. Existing digital archives (ASI, DLI, eGangotri) provide scanned images but
no automated transcription layer. This project addresses that gap by building an
automated AI pipeline that makes previously unreadable artefacts legible and searchable.

### Goal statement (quote directly in report/paper)
> "Take unclear, degraded images of stone inscriptions, palm leaf manuscripts,
> copper plates, paper manuscripts, and cave/rock paintings — and produce clean,
> readable, searchable, citable digital records for researchers, historians,
> linguists, and the public."

### Scope — Phase 1 (current, completed)
The project delivers **Stages 1–4** of the pipeline:
1. Preprocessing
2. AI Enhancement
3. Binarisation
4. OCR & Transcription ← **Phase 1 endpoint**

Translation (Stage 5) is architecturally designed and model-selected but deferred
to **Phase 2 (time-permitting)**, per mentor guidance to first deliver a robust,
well-evaluated OCR system.

### Output per processed artefact
- Enhanced image (noise-reduced, super-resolved, colour-corrected)
- Unicode transcription of the original script
- Structured JSON record with full metadata
- Exportable PDF research record with citation block

---

## 2. Supported Artefact Types

Five artefact categories are in scope. Each has distinct degradation characteristics
requiring different algorithmic treatment. This is important context for both the
system design section (report) and the problem statement (paper).

| Artefact Type | Primary Degradation | Key Algorithm Used |
|---|---|---|
| Stone inscriptions | Low contrast, surface weathering, shadow | DStretch colour enhancement + Real-ESRGAN |
| Palm leaf manuscripts | Yellowing, ink fading, physical fragility | Binarisation + contrast stretch |
| Copper plate inscriptions | Reflective surface, oxidation patina | HDR normalisation + Real-ESRGAN |
| Paper manuscripts | Foxing, stains, torn edges | LaMa inpainting + denoising |
| Cave / rock paintings | Uneven lighting, rough irregular texture | DStretch + shadow removal |

**Key point for paper:** The multi-artefact design is a deliberate generalist approach.
Rather than a single-type specialist system, the pipeline routes each image through
artefact-specific processing chains, making it broadly applicable across South Asian
heritage institutions.

---

## 3. System Architecture

### End-to-end pipeline

```
Raw scanned image input (JPG / TIFF / PNG)
          ↓
Stage 1 — Preprocessing        (normalise, white balance, crop, orient)
          ↓
Stage 2 — AI Enhancement       (denoise, sharpen, super-resolution, DStretch)
          ↓
Stage 3 — Binarisation         (separate text pixels from background)
          ↓
Stage 4 — OCR / Transcription  (extract characters as Unicode text)    ← Phase 1 endpoint
          ↓
Stage 5 — Translation          (ancient script → modern English)        ← Phase 2 (future)
          ↓
Stage 6 — Record Assembly      (bundle image + text + metadata → JSON)
          ↓
Stage 7 — Storage & Export     (JSON database, PDF export)
          ↓
Stage 8 — Web UI               (React + FastAPI — browse, process, compare) ✅ Implemented
```

### Technology stack (for report system design section)

| Layer | Technology | Version |
|---|---|---|
| Image processing core | OpenCV | 4.9.0.80 |
| Image manipulation | Pillow | 10.3.0 |
| AI super-resolution | Real-ESRGAN (basicsr) | 0.3.0 / 1.4.2 |
| Deep learning runtime | PyTorch | ≥ 2.0.0 |
| OCR engine 1 | Tesseract + pytesseract | 0.3.10 |
| OCR engine 2 | EasyOCR | 1.7.1 |
| Indic transliteration | indic-transliteration | 2.3.57 |
| NLP / Translation (Phase 2) | Hugging Face Transformers | 4.40.0 |
| Web backend | FastAPI + Uvicorn | 0.111.0 / 0.29.0 |
| Web frontend | React 19 + Vite 6 | — |
| Frontend styling | Tailwind CSS v4 | — |
| Frontend data-fetching | TanStack Query v5 | — |
| PDF export | fpdf2 | 2.7.9 |
| Storage | TinyDB + SQLAlchemy | 4.8.0 / 2.0.30 |
| Numerical computing | NumPy | 1.26.4 |
| Scientific image ops | scikit-image | 0.23.2 |

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

**Output format:** JPEG (quality=95). TIFF was considered but rejected — see
Section 6, Key Decisions.

---

### Stage 2 — AI Enhancement (`src/enhance.py`)

**Purpose:** Core value-add of the project. Makes previously unreadable inscriptions
legible using AI super-resolution and colour science.

**Functions:**
- `denoise(img, strength=10)` — Non-local means denoising.
  `strength=10` for mild degradation, `20` for heavy damage.
- `enhance_with_realesrgan(img, scale=2, model_path=...)` — Super-resolution and
  sharpening. Uses `outscale=2` (from a 4x model) to avoid over-smoothing.
- `dstretch(img, colour_space="LAB")` — DStretch decorrelation stretch. Computes
  covariance of colour channels, removes inter-channel correlation via eigendecomposition
  to reveal colour differences invisible to the human eye. Particularly powerful for
  cave paintings and faded stone. Based on algorithm by Jon Harman (dstretch.com).
  Colour space options: LAB, YDS, YBK, LDS.
- `sharpen(img, amount=1.5)` — Unsharp mask sharpening to crisp character edges.
- `enhance(img_path, output_path, use_dstretch=False)` — Full chain.

**Per-artefact processing chains:**
- Stone inscriptions: denoise → Real-ESRGAN → sharpen
- Cave paintings: denoise → DStretch → sharpen
- Palm leaf manuscripts: auto_white_balance → denoise → Real-ESRGAN

**Real-ESRGAN configuration:**
```
RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
tile=400, tile_pad=10, half=False
```
Tiling prevents out-of-memory errors on large images.

---

### Stage 3 — Binarisation (`src/binarise.py`)

**Purpose:** Convert enhanced colour image to clean binary (black text, white background)
for OCR.

**Functions:**
- `binarise_sauvola(img, window_size=25)` — Sauvola local adaptive thresholding.
  **Preferred method** for most inscription types. Handles spatially uneven backgrounds
  (stone texture, aged palm leaf) better than global methods.
- `binarise_otsu(img)` — Otsu global thresholding. Fast; suitable for clean paper
  manuscripts with uniform backgrounds.
- `binarise_adaptive(img)` — OpenCV adaptive mean thresholding. Fallback for mixed
  quality images.
- `remove_noise_blobs(binary, min_size=50)` — Removes small disconnected components
  (dust, surface damage artefacts) from binary image.
- `binarise(img_path, output_path, method="sauvola")` — Main entry point.

**Post-binarisation step:** Morphological closing (`cv2.MORPH_CLOSE`, 2×2 kernel)
reconnects broken character strokes — important for weathered inscriptions where
character strokes are partially eroded.

**Output format:** PNG (lossless — no compression artefacts on binary data).

---

### Stage 4 — OCR & Transcription (`src/ocr.py`)

**Purpose:** Extract characters from the binarised image as Unicode text.

**Supported scripts and routing:**

| Script | Tesseract language | EasyOCR language |
|---|---|---|
| Tamil | `tam` | `ta` |
| Sanskrit | `san` | `hi` (Devanagari fallback) |
| Kannada | `kan` | `kn` |
| Telugu | `tel` | `te` |
| Malayalam | `mal` | `ml` |
| Devanagari | `hin` | `hi` |
| Brahmi | — (custom model needed) | — |
| Grantha | — (custom model needed) | — |

**Ensemble approach:** Both Tesseract and EasyOCR are run independently; results
are merged by confidence score. This is a deliberate design decision — neither engine
is individually reliable for ancient Indic scripts.

**Tesseract config:** `--oem 1 --psm 6` (LSTM engine, uniform block of text)
**EasyOCR config:** `detail=1` for per-word bounding boxes

**Confidence thresholds:**
- ≥ 0.85 → marked as **verified**
- 0.60–0.84 → marked as **review needed**
- < 0.60 → marked as **uncertain** (highlighted in output, flagged for manual review)

**Transcription output schema:**
```json
{
  "script": "tamil",
  "text": "கஞ்சி மாநகர் பல்லவ குல தீபம்",
  "lines": [
    {
      "line_number": 1,
      "text": "கஞ்சி மாநகர்",
      "confidence": 0.91,
      "bounding_box": [x, y, w, h],
      "uncertain": false
    }
  ],
  "overall_confidence": 0.87,
  "engine_used": "tesseract+easyocr ensemble",
  "uncertain_regions": [[x1, y1, x2, y2]]
}
```

**Brahmi and Grantha scripts:** No off-the-shelf OCR model exists for these ancient
scripts. These are flagged for manual transcription in Phase 1. A custom model using
the Brahmi character dataset (1032 Ashokan Brahmi characters, 258 classes;
arxiv.org/abs/2501.01981) is planned for Phase 2.

---

### Stage 5 — Translation (`src/translate.py`) — PHASE 2 (NOT YET IMPLEMENTED)

> **This stage is out of scope for Phase 1.** Architecture and model selection are fully
> designed. In all Phase 1 records, the translation field is `null` and status is
> `"phase_2_pending"`. No schema change will be required when Phase 2 is implemented.

**Planned model routing:**
- Post-10th century CE texts: Helsinki-NLP OPUS-MT models
  (`Helsinki-NLP/opus-mt-dra-en` for Dravidian scripts,
  `Helsinki-NLP/opus-mt-hi-en` for Hindi/Sanskrit)
- Ancient / classical texts (pre-10th century CE): LLM fallback (Claude / GPT-4)
  with artefact context provided as system prompt

**Planned output includes:** English translation, modern source-language version
(e.g. classical Tamil → modern Tamil), confidence score, translator notes for
ambiguous segments, and preserved proper nouns in original script + romanisation.

---

### Stage 6 — Record Assembly (`src/record.py`)

**Purpose:** Bundle all pipeline outputs into a single structured, citable research record.

**Record ID format:** `INS-{YYYY}-{NNNN}` (e.g. `INS-2024-0047`)

**Record fields (key fields for report/paper):**
- `record_id`, `created_at`, `status` (draft / review / verified)
- `artefact`: type, material, period, dynasty, location (with coordinates), dimensions,
  condition, collection name, accession number
- `images`: paths to original, enhanced, binarised, thumbnail; enhancement method;
  processing timestamp
- `transcription`: script, full Unicode text, per-line breakdown, confidence scores,
  OCR engine used
- `translation`: `null` in Phase 1; fully populated in Phase 2
- `citation`: suggested citation string, DOI (future), CC BY 4.0 licence
- `processing_log`: per-stage duration and status

**Export:** `export_pdf()` generates a researcher-friendly PDF with side-by-side
image comparison, transcription, metadata table, and citation block using `fpdf2`.

---

### Stage 8 — Web UI (`api/` + `web/`) ✅ Implemented April 2026

**Stack:** FastAPI (port 8000) + React 19 + Vite 6 + Tailwind CSS v4 + TanStack Query v5.

**Note for writing:** The original architecture used Gradio. This was replaced with
React + FastAPI for richer interactivity — specifically the before/after comparison
slider, async job polling, and sidebar layout. The Gradio stub is retained in the
codebase for reference only.

**Backend API endpoints:**
- `GET /api/images` — list all images in `data/raw/`
- `POST /api/process` — submit image(s) for pipeline processing, returns job ID
- `GET /api/jobs/{id}` — poll job status and retrieve results
- Static file serving for enhanced/binarised images

**Frontend components:** `ImageGrid`, `ImageCard`, `StagePanel`, `ResultViewer`,
`ComparisonSlider`, `ProgressBar`

**React hooks:** `useImages` (TanStack Query), `useJob` (polling)

**UI tabs:**
1. Process new image — upload, fill metadata, run pipeline, view record
2. Browse records — searchable gallery with thumbnails
3. View record — before/after slider, transcription display
4. Export — download as PDF or JSON
5. Translation tab — **disabled in Phase 1, enabled in Phase 2**

---

## 5. Interdisciplinary Team Roles

This project is built by a three-discipline team. Each branch has exclusive ownership
of specific components. All three contributions are required for project completion.

> **Note:** The project works entirely on already-scanned existing images. Hardware
> acquisition, camera rigs, and field capture are explicitly out of scope.

### CS / IT — AI Pipeline & Software

Owns the end-to-end software from raw image to final record.

| Responsibility | File |
|---|---|
| Preprocessing | `src/preprocess.py` |
| AI enhancement (Real-ESRGAN, DStretch) | `src/enhance.py` |
| Binarisation | `src/binarise.py` |
| OCR & transcription | `src/ocr.py` |
| Translation layer (Phase 2) | `src/translate.py` |
| Record assembly & PDF export | `src/record.py` |
| Pipeline orchestration | `src/pipeline.py` |
| FastAPI backend | `api/main.py`, `api/jobs.py`, `api/pipeline.py` |
| React frontend | `web/src/` |

---

### ECE — Signal Processing & Image Quality Analysis

Owns the analytical rigour of the image processing stages. Brings signal processing
theory to improve and validate pipeline outputs. No hardware component.

**Contribution 1 — Noise characterisation & modelling**
Profiles noise types present in scanned inscription images across all five artefact
categories: Gaussian (scanner sensor), salt-and-pepper (dust/damage), JPEG compression
artefacts, and periodic noise (scanner line artefacts). Deliverable: `docs/noise_analysis_report.pdf`

**Contribution 2 — Custom filter design (`src/filters.py`)**
- `gabor_filter_bank(img, frequencies=[0.1,0.2,0.4], orientations=8)` — Separates
  inscription texture from background using oriented spatial frequency filters
  (Daugman 1985).
- `directional_edge_enhance(img, angle_deg=45.0)` — Enhances carving edges in a
  specified direction.
- `remove_periodic_noise_fft(img, threshold=0.1)` — FFT-based scanner line artefact
  removal. These filters are called from `src/enhance.py` as optional steps selectable
  per artefact type.

**Contribution 3 — Histogram & colour channel analysis (`src/analysis.py`)**
- Per-channel statistical analysis (mean, std, skewness, kurtosis) across artefact types.
- DStretch colour space parameter tuning (LAB vs YDS vs YBK vs LDS) per material.
- Before/after histogram plots for the project report.

**Contribution 4 — Image quality metrics (`src/metrics.py`)** — owns the evaluation layer
- `compute_psnr(original, enhanced)` — Peak signal-to-noise ratio. Target ≥ 30 dB.
- `compute_ssim(original, enhanced)` — Structural similarity index. Target ≥ 0.85.
- `compute_cnr(img, text_mask)` — Contrast-to-noise ratio between text and background.
  Particularly relevant for low-contrast stone inscriptions.
- `compute_sharpness(img)` — Laplacian variance as edge sharpness proxy.
- `full_quality_report(original, enhanced, text_mask)` — Consolidated quality dict;
  integrated into `src/pipeline.py` so every processed image is automatically scored.

---

### IEM — Process Design, Project Management & Impact Analysis

Owns the operational and organisational layer.

**Contribution 1 — Project management**
Gantt chart, weekly sprint planning, task tracking, risk register (risks include: OCR
accuracy below threshold, dataset unavailability, compute constraints).
Deliverable: `docs/project_plan.xlsx` (updated weekly)

**Contribution 2 — Digitisation workflow design**
Value stream mapping of the pipeline to identify bottlenecks (e.g., OCR stage is
~3× slower than enhancement). Proposes parallelisation and batching strategies.
Throughput measurement: inscriptions processed per hour.
Deliverable: `docs/workflow_analysis.pdf`

**Contribution 3 — Cost-benefit & scalability analysis**
- Compute cost per inscription (cloud GPU vs local CPU)
- Storage cost per inscription (TIFF master + access copies + records)
- Researcher time saved vs manual transcription
- Scalability model: 100 → 10,000 → 1,000,000 records
Deliverable: `docs/cost_benefit_analysis.pdf`

**Contribution 4 — Stakeholder documentation**
Non-technical project overview, user guide for the UI, data governance policy
(ownership, access control, CC BY 4.0 licensing), impact statement for ASI /
museum curators / government bodies.
Deliverables: `docs/user_guide.pdf`, `docs/impact_statement.pdf`

---

### Ownership summary table

| Component | Owner |
|---|---|
| AI enhancement pipeline | CS/IT |
| OCR & transcription | CS/IT |
| Translation layer | CS/IT |
| Web UI / FastAPI portal | CS/IT |
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
| Sanskrit OCR Dataset | Classical Sanskrit document images | github.com/ihdia/sanskrit-ocr |
| Brahmi Character Dataset | 1032 Ashokan Brahmi characters, 258 classes | arxiv.org/abs/2501.01981 |
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

| Metric | What it measures | Tool | Target threshold |
|---|---|---|---|
| PSNR | Signal-to-noise ratio after enhancement | `skimage.metrics.peak_signal_noise_ratio` | ≥ 30 dB |
| SSIM | Structural similarity to ground truth | `skimage.metrics.structural_similarity` | ≥ 0.85 |
| OCR confidence | Average word confidence from OCR engine | Tesseract / EasyOCR built-in | ≥ 0.70 |
| CER | Character error rate vs. ground truth | `jiwer` library | ≤ 0.15 |
| CNR | Contrast-to-noise ratio (text vs background) | `src/metrics.py` (ECE) | — |
| Edge sharpness | Laplacian variance | `src/metrics.py` (ECE) | Higher = better |

### Reporting these in the paper
For the Results section, report mean ± std of each metric across the test set,
broken down by artefact type (stone / palm leaf / copper / paper / cave).
This demonstrates the pipeline's generalisation across material types.

### Qualitative evaluation
The React UI's before/after comparison slider is the primary tool for qualitative
human evaluation. Include representative before/after image pairs in both the report
and paper figures.

---

## 8. Key Design Decisions & Rationale

Use this section in the report (Design Choices) and paper (Methodology / Related Work).

| Decision | Rationale |
|---|---|
| Real-ESRGAN over bicubic upscaling | Trained on real-world degraded photos; preserves texture and character stroke edges far better than classical upscaling |
| DStretch for rock / cave art | Originally developed for rock art analysis by Jon Harman; reveals colour differences invisible to the human eye through decorrelation stretch |
| Sauvola over Otsu binarisation | Handles spatially uneven backgrounds (stone surface, aged palm leaf fibre) better than global threshold methods |
| EasyOCR + Tesseract ensemble | Neither engine alone is reliable for ancient Indic scripts; combining raises overall confidence and recall |
| JSON record format | Human-readable, version-controllable, trivially exportable to any downstream format (XML, CSV, RDF) |
| React + FastAPI (replaced Gradio) | Enables richer interactivity: before/after comparison slider, async job polling, sidebar layout — not achievable in Gradio |
| JPEG (quality=95) for preprocessing output | Avoids PIL/libtiff metadata write failures on Windows; smaller files; quality=95 is sufficient for all subsequent pipeline stages |
| `outscale=2` for Real-ESRGAN | Uses 4× model but outputs at 2× resolution — avoids over-smoothing and hallucinated detail on dense ancient script |
| Phase 1 endpoint at OCR | Mentor guidance: deliver robust, well-evaluated transcription before translation; prevents translation errors compounding OCR errors |

---

## 9. Implementation Phases & Timeline

| Phase | Deliverable | Duration | Status |
|---|---|---|---|
| Phase 1 | Environment setup, folder structure, 5 sample images tested | Week 1 | ✅ Complete |
| Phase 2 | `preprocess.py` + `enhance.py` — full enhancement pipeline with batch processing | Weeks 2–3 | ✅ Complete |
| Phase 3 | `binarise.py` + `ocr.py` — binarisation and OCR for Tamil and Sanskrit | Weeks 4–6 | ✅ Complete |
| Phase 4 | `translate.py` — translation layer | Weeks 7–9 | 🔲 Phase 2 (time-permitting) |
| Phase 5 | `record.py` — record assembly, JSON storage, PDF export | Weeks 10–11 | ✅ Complete (translation field null) |
| Phase 6 | Web UI (React + FastAPI) — browse, process, compare | Weeks 12–14 | ✅ Complete (April 2026) |

---

## 10. Non-Destructive Processing Policy

These are mandatory system constraints — include in the report's system design section
and cite in the paper as part of the archival preservation methodology.

1. **Never overwrite originals.** `data/raw/` is read-only. All outputs go to
   `data/enhanced/`, `data/binarised/`, etc.
2. **3-2-1 backup rule:** 3 copies of original data, on 2 different storage media,
   with 1 offsite/cloud backup.
3. **Preserve EXIF metadata** through all processing stages.
4. **Log every operation** — stage, timestamp, pipeline version, model weights version.
5. **Version control model weights** — every record logs exactly which model version
   was used to produce its enhanced image.

---

## 11. Limitations & Future Work

Use this section for the Conclusion / Future Work of both documents.

### Current limitations

- **Translation deferred to Phase 2.** The architecture is fully designed. Standard
  MT models (Helsinki-NLP OPUS-MT) are selected for post-10th century CE texts;
  LLM fallback (Claude / GPT-4) is planned for archaic classical forms.

- **Brahmi and Grantha scripts unsupported for OCR.** No off-the-shelf model exists.
  These are flagged for manual transcription. A custom model using the 1032-character
  Brahmi dataset is planned.

- **Heavy damage threshold.** Images with >50% surface damage produce low-confidence
  OCR. A planned fix: inpainting using LaMa (Large Mask Inpainting) before OCR.

- **3D inscriptions.** Deep-carved stone inscriptions benefit from 3D/LiDAR data.
  The Kaggle dataset includes 3D models; processing 3D point clouds is future work.

- **Multi-script inscriptions.** Some artefacts combine two scripts (e.g. Tamil +
  Grantha). Future work: segment by script region before OCR routing.

### Future work

- Phase 2 translation layer (Helsinki-NLP MT models + LLM fallback)
- Custom OCR models for Brahmi and Grantha scripts
- LaMa inpainting for heavily damaged images
- 3D point cloud processing pipeline for carved inscriptions
- Multi-script region segmentation
- Public-facing web portal (Omeka S or custom React frontend) as Phase 7

---

## 12. References

Cite these in both the report and the research paper.

- **Real-ESRGAN:** Wang, X. et al. (2021). Real-ESRGAN: Training Real-World Blind
  Super-Resolution with Pure Synthetic Data. arXiv:2107.10833
- **DStretch:** Harman, J. (2008). Using Decorrelation Stretch to Enhance Rock Art
  Images. dstretch.com
- **SSIM metric:** Wang, Z. et al. (2004). Image quality assessment: from error
  visibility to structural similarity. *IEEE Trans. Image Processing*, 13(4), 600–612.
- **Gabor filters:** Daugman, J. G. (1985). Uncertainty relation for resolution in
  space, spatial frequency, and orientation optimised by two-dimensional visual
  cortical filters. *JOSA A*, 2(7), 1160–1169.
- **Value stream mapping:** Rother, M. & Shook, J. — *Learning to See*
  (Lean Enterprise Institute)
- **Indiscapes dataset:** IIIT Hyderabad — ihdia.iiit.ac.in
- **Brahmi OCR dataset:** arXiv:2501.01981
- **Tamil NLP resources:** AI4Bharat — ai4bharat.org
- **Indic OCR models:** Bhashini — bhashini.gov.in
- **AWS pricing calculator:** calculator.aws (for cost-benefit analysis)

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
│   ├── binarised/               ← Stage 3 output
│   ├── transcriptions/          ← Stage 4 text output
│   ├── translations/            ← Stage 5 output (Phase 2)
│   └── records/                 ← final JSON records
├── models/
│   └── weights/                 ← RealESRGAN_x4plus.pth, RealESRGAN_x4plus_anime_6B.pth
├── api/                         ← FastAPI backend ✅
│   ├── main.py                  ← /api/images, /api/process, /api/jobs/{id}
│   ├── jobs.py                  ← thread-safe in-memory job store
│   └── pipeline.py              ← adapter to src/preprocess.py
├── web/                         ← React + Vite frontend ✅
│   └── src/
│       ├── App.tsx
│       ├── types.ts
│       ├── api/client.ts
│       ├── hooks/
│       └── components/
├── src/
│   ├── preprocess.py            ← Stage 1 ✅
│   ├── enhance.py               ← Stage 2
│   ├── binarise.py              ← Stage 3
│   ├── ocr.py                   ← Stage 4
│   ├── translate.py             ← Stage 5 (Phase 2)
│   ├── record.py                ← Stage 6
│   ├── pipeline.py              ← orchestrator
│   ├── filters.py               ← ECE: Gabor, FFT, directional filters
│   ├── analysis.py              ← ECE: colour distribution tools
│   ├── metrics.py               ← ECE: PSNR, SSIM, CNR, sharpness
│   └── utils.py                 ← shared helpers ✅
├── docs/
│   ├── noise_analysis_report.pdf     ← ECE deliverable
│   ├── project_plan.xlsx             ← IEM deliverable
│   ├── workflow_analysis.pdf         ← IEM deliverable
│   ├── cost_benefit_analysis.pdf     ← IEM deliverable
│   ├── user_guide.pdf                ← IEM deliverable
│   └── impact_statement.pdf         ← IEM deliverable
├── tests/
│   ├── test_preprocess.py       ← ✅ 4 tests
│   ├── test_api.py              ← ✅ 11 tests
│   ├── test_enhance.py
│   └── test_ocr.py
└── outputs/
    ├── exports/                 ← PDF exports
    └── logs/                    ← processing logs
```

---

*Last updated: May 2026.*
*Phase 1 (Stages 1–4) is complete. Phase 2 (Translation) is time-permitting.*
*Per mentor guidance (March 2026): deliver robust OCR before proceeding to translation.*
*Agent or writer: read this file top-to-bottom before drafting any section.*
