# Colonoscopy Feature Matching Evaluation

This project provides an evaluation of state-of-the-art feature matchers for monocular colonoscopy pose recovery, using an incremental Structure-from-Motion (SfM) style pipeline.

It supports evaluating multiple learned feature matchers and ensembles on the C3VD phantom dataset, comparing their effects on pose estimation and reconstruction metrics.

> **Note:** This project is a research prototype and is not packaged as a ready-to-use software library.

---

## Project Structure

```text
thesis-colonoscopy-eval/
├── ablation_outputs/               # Outputs of ablation data analysis: plots etc
├── data/
│   └── C3VD/cecum_t1_a/            # Example undistorted video sequence
├── data_analysis_outputs/          # Outputs of main data analysis: plots etc
├── env/
│   └── environment.yml             # Conda environment file
├── libs/                           # Custom libraries (if used)
├── notebooks/
│   └── matching_analysis.ipynb     # Main experimental development 
├── results/                        # Output folder for experiment results
├── scripts/
│   └── matching_analysis.py        # Main analysis script
├── src/
│   ├── ensemble.py                 # Ensemble matcher logic
│   ├── io.py                       # Data loading and I/O utilities
│   ├── pose_utils.py               # Pose estimation utilities
├── ablation_run.sh                 # Example run script
├── ablation_study_modified.ipynb   # Analysis notebook of ablation study
├── data_analysis.ipynb             # Analysis notebook of main matchers study
├── run_experiments.sh              # Example run script
├── batch_process.sh                # Example run script
├── debug_single_video.sh           # Example debug script
├── README.md

```

---

## Data

- Example sequence provided in:
  - `data/C3VD/cecum_t1_a/`
  - Includes undistorted RGB frames (`*_color.png`), depth maps (`*_depth.tiff`), mesh, and ground truth poses.

---

## Running the Project

### 1. Setup Environment

It is recommended to use Conda:

```bash
conda env create -f environment.yml
conda activate my_env
```

> **Note:** No `requirements.txt` is provided — use the supplied `environment.yml`.

---

### 2. Running Experiments

Example run commands:

```bash
bash run_experiments.sh
bash ablation_run.sh
bash debug_single_video.sh
```

These scripts will run matcher evaluation on the included C3VD example video.

---

### 3. Main Analysis

The primary script is:

```bash
python scripts/matching_analysis.py
```

This performs feature matching, pose estimation, and outputs metrics.

---

## Notes and Limitations

- The project is **not a full reconstruction pipeline** — global optimisation (bundle adjustment, loop closure) is not included.
- The evaluation is **prototype code** — some paths and structure may need adjustment depending on your system.
- The screening video evaluation is incomplete — see thesis Discussion.
- The phantom data lacks non-rigid deformation; real-world performance will be lower.

---

## Future Improvements

- Combine with pre-processing (specularity removal, occlusion filtering)
- Integrate global optimisation and full SLAM pipeline
- Evaluate on real clinical data
- Improve project structure and packaging

---

## Citation

If you use this project, please cite:

> **Honor Duthie**,  
> *Evaluation of Learned Feature Matchers for Monocular Colonoscopy Pose Recovery*,  
> MSc Thesis, Universitat Pompeu Fabra, 2025.
