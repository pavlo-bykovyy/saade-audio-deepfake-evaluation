# SAADE–AUA — Scenario-Based Testing of Synthetic Speech Detectors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](workflows/)
[![Dataset: ATV-UA](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ATV--UA-FFD21E)](https://huggingface.co/datasets/adam-i/ATV-UA)
[![Models: AASIST | Wav2Vec2](https://img.shields.io/badge/Models-AASIST%20%7C%20AASIST--L%20%7C%20Wav2Vec2-blueviolet)](#main-notebooks-current-code)
[![Scenarios: 21](https://img.shields.io/badge/Acoustic%20scenarios-21-informational)](#main-notebooks-current-code)
[![Citation: CFF](https://img.shields.io/badge/Cite-CITATION.cff-brightgreen)](CITATION.cff)
<!-- After the Zenodo release, add the DOI badge, e.g.:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

*Українська версія: [README_UA.md](README_UA.md)*

Source code for the research project **Scenario-Adaptive Audio Deepfake Evaluation (SAADE)**. The main goal is **to evaluate pretrained models across acoustic scenarios, not to train models**. The repository contains the final research workflows v9/v9.1/v9.2 and all available earlier versions from the project history.

**Dataset:** [ATV-UA (Hugging Face)](https://huggingface.co/datasets/adam-i/ATV-UA). The research server also used a local AUA directory structure. Do not assume that the published ATV-UA release is byte-identical to the local AUA without comparing manifests.

## Main notebooks (current code)

| Stage | Code | Purpose |
|---|---|---|
| REAL–TTS preparation | `workflows/01_generate_matched_tts/` | Builds text-matched pairs; Piper generation |
| Full evaluation | `workflows/02_full_scenario_evaluation/` | All usable canonical AUA plus public REAL–TTS pairs, 3 models × 21 scenarios |
| Model reference parity | `workflows/03_aasist_reference_parity/` | ONNX ↔ PyTorch comparison and padding effect |
| AASIST correction | `workflows/04_repeat_padding_correction/` | Reference repeat-padding and recalibration of AASIST/AASIST-L |
| FREEZE / hidden protocol | `workflows/05_freeze_hidden_once/` | Freezes parameters and protects the one-time hidden test |

**The main experimental result is v9.1 + v9.2.** The original v9 used zero-padding for AASIST; v9.1 corrects this without re-running Wav2Vec2. v9.2 applies the frozen configuration to the one-time hidden test. Older versions in `historical/` are **a development archive and are not recommended for reproducing the final study**. In particular, v4–v6 contain E1/E2/E3 *training* experiments; that is a different setup, not the final published methodology.

## Structure

- `workflows/` — the five current steps, each run as a separate `.ipynb`.
- `dataset_audits/` — independent audit and diagnostics of AUA metadata.
- `historical/` — all preserved code releases v1–v8, fixes, and early v9 iterations.
- `paper/tools/` — Python scripts for preparing the quantitative figures and the manuscript.
- `docs/` — reproducibility, data structure, history, and GitHub instructions (in Ukrainian).
- `requirements.txt` — core libraries without pinned versions; historical releases have their own `requirements_*.txt`.

## Running

1. Create a Python 3.11 / JupyterLab environment. Install `requirements.txt` and system `ffmpeg`; Piper requires separate voice models and TTS dependencies.
2. Place input data in `/home/jovyan/work/audio_real_fake/AUA` and working results in `/home/jovyan/work/audio_real_fake/results`. Paths are currently set **inside the notebooks** and must be adjusted for another server.
3. Run `workflows/` in the order listed; v9.1 requires completed v9 results. Pretrained model weights are downloaded separately under the terms of their respective sources.
4. Do not open the hidden test for debugging: `FREEZE → HIDDEN_ONCE` must be run once, and only after the protocol is final.

## Data and publication

**Not included in this archive**: source audio recordings, TTS WAVs, model checkpoints, private split manifests, 876,330 segment-level predictions, internal logs, and hidden test results. This is source code for GitHub, **not a backup of the results**.

The code and protocols were created as part of the SAADE–AUA study. **License: [MIT](LICENSE)** — applies only to the source code in this repository. The dataset, pretrained models (AASIST, AASIST-L, Wav2Vec2), Piper voices, and other third-party components are subject to their own licenses and terms of use.

See `docs/REPRODUCIBILITY_UA.md` and `docs/GITHUB_PUBLISHING_UA.md`.

## Authors

- **[Khrystyna Lipianina-Honcharenko](https://github.com/Khrystyna-LH)** — Core Developer
- **[pavlo-bykovyy](https://github.com/pavlo-bykovyy)** — Core Developer
- **[Oleh Honcharenko]** — Co-author & Developer

West Ukrainian National University (WUNU)

## Citation

If you use this code in your research, please cite it. Citation metadata is provided in [`CITATION.cff`](CITATION.cff); on GitHub, use the **Cite this repository** button in the sidebar to get APA or BibTeX.

```bibtex
@software{lipianina_honcharenko_2026_saade_aua,
  author    = {Lipianina-Honcharenko, Khrystyna and Bykovyy, Pavlo and Honcharenko, Oleh},
  title     = {{SAADE-AUA: Scenario-Adaptive Audio Deepfake Evaluation on Ukrainian Speech}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.1},
  license   = {MIT}
}
```

A DOI will be added after the release is archived on Zenodo.
