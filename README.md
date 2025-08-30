# A Reproducible Benchmark of Visual Attention Mechanisms in CNNs

This project provides a professional, end-to-end Python pipeline to investigate the impact of modern visual attention mechanisms (Squeeze-and-Excitation, CBAM) against a standard ResNet-style baseline. The benchmark is run across four diverse datasets to test for generalization.

The entire system is built with MLOps best practices in mind, featuring modular code, command-line configuration, and comprehensive experiment tracking with MLflow.

## Key Findings: The Impact of Attention

The experiments reveal that attention mechanisms provide a significant, task-dependent performance boost, especially on complex and fine-grained datasets.

| Dataset          | Baseline (None) | SE (Squeeze-and-Excitation) | CBAM (Convolutional Block) |
| :--------------- | :-------------: | :-------------------------: | :------------------------: |
| CIFAR10          | 86.97%          | **87.01%** (+0.04%)         | 85.89% (-1.08%)            |
| CIFAR100         | 57.26%          | **59.28%** (+2.02%)         | 58.07% (+0.81%)            |
| Oxford102Flowers | 28.13%          | 37.53% (+9.40%)             | **38.74%** (+10.61%)       |
| EuroSAT          | 97.41%          | **97.61%** (+0.20%)         | 97.37% (-0.04%)            |

## Tech Stack & Skills Demonstrated
- **Frameworks:** Python, PyTorch, Torchvision
- **MLOps:** MLflow (for experiment tracking), `argparse` (for configuration management)
- **Engineering Practices:** Modular & Reusable Code Structure, GPU Memory Management, Object-Oriented Programming (OOP), Reproducible Research Methodology.
- **Tools:** Git, Bash, Jupyter (for initial exploration)

## How to Reproduce the Results

This project is structured for easy reproduction on any machine with a CUDA-enabled GPU.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chillguycode/visual_attention_benchmarking.git
    cd visual_attention_benchmarking
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run a training experiment:**
    The `run_experiment.py` script is the main entry point. All parameters are configurable via the command line.
    ```bash
    # Example: Run the SE model on the CIFAR100 dataset for 30 epochs
    python run_experiment.py --dataset CIFAR100 --attention_type se --epochs 30
    ```
    ```bash
    # Example: Run the baseline model on EuroSAT
    python run_experiment.py --dataset EuroSAT --attention_type none
    ```

4.  **View and Compare All Experiment Results:**
    All parameters and metrics are automatically logged with MLflow. To launch the tracking dashboard and compare your runs, execute:
    ```bash
    mlflow ui
    ```
    Then navigate to `http://127.0.0.1:5000` in your browser.
