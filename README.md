fast-rich-simulation/          # Main project directory \
│ \
├── config/                    # Configuration files (keep parameters out of code) \
│   └── default.yaml          # Main config: paths, hyperparams, physics constants \
│ \
├── data/                     # ALL data lives here (DO NOT commit to Git) \
│   ├── raw/                  # (Optional) Placeholder for any raw data \
│   ├── processed/            # Your generated .npz datasets go here \
│   └── external/             # (Optional) Placeholder for 3rd party data \
│ \
├── docs/                     # Documentation \
│   └── project_specification.md # Your planning notes, physics formulas, etc. \
│ \
├── models/                   # Trained model weights \
│   ├── checkpoints/          # PyTorch Lightning or regular checkpoints \
│   └── best_model.pth        # The final trained model state dict \
│ \
├── notebooks/                # For exploration and presentation \
│   ├── 01_simulator_validation.ipynb \
│   ├── 02_data_exploration.ipynb \
│   └── 03_final_results.ipynb \
│ \
├── src/                      # The main source code (The core of your project) \
│   ├── __init__.py           # Makes `src` a Python module \
│   │ \
│   ├── physics/              # Module for the detector simulation \
│   │   ├── __init__.py \
│   │   ├── rich_simulator.py # Contains the RichSimulator class \
│   │   └── constants.py      # Particle masses, refractive indices, etc. \
│   │
│   ├── ml/                   # Module for the machine learning \
│   │   ├── __init__.py \
│   │   ├── diffusion/        # Sub-module for the diffusion model \
│   │   │   ├── __init__.py \
│   │   │   ├── unet.py       # U-Net architecture definition \
│   │   │   ├── scheduler.py  # The noise scheduler (beta schedule) \
│   │   │   └── diffusion.py  # The core DDPM class (training & sampling logic) \
│   │   │ \
│   │   └── utils/            # For the PID validation \
│   │       ├── __init__.py \
│   │       └── pid_classifier.py # Your old PID model, adapted for this project \
│   │ \
│   ├── data/                 # Module for data handling \
│   │   ├── __init__.py \
│   │   ├── dataset.py        # PyTorch Dataset class for RICH images \
│   │   └── generate_data.py  # Script to create the dataset using the simulator \
│   │ \
│   ├── utils/                # General utilities \
│   │   ├── __init__.py \
│   │   ├── logging.py        # Setup for logging \
│   │   ├── visualization.py  # Functions for plotting images, rings, etc. \
│   │   └── metrics.py        # KS test, calculating separation power, etc. \
│   │ \
│   └── scripts/              # Runnable scripts for main tasks \
│       ├── generate_data.py  # Calls src.data.generate_data \
│       ├── train.py          # Script to train the diffusion model \
│       └── validate.py       # Script to run the full validation pipeline \
│ \
├── tests/                    # Unit tests (Very impressive to have!) \
│   ├── __init__.py \
│   ├── test_simulator.py     # Tests for the physics simulator \
│   └── test_diffusion.py     # Tests for the diffusion model components \
│ \
├── requirements.txt          # Python dependencies \
├── environment.yml           # (Optional) For Conda users \
├── pyproject.toml           # (Optional) Modern package config \
├── .gitignore               # Crucial: ignores data/, models/, etc. \
└── README.md                # Your project's front page - MUST be excellent \