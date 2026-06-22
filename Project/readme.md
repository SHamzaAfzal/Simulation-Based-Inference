# Inference of Protein Secondary Structure Motifs using Simulation-Based Inference

## Overview
This repository contains the implementation and findings for the project "Inference of Protein Secondary Structure Motifs". The primary goal is to study the per-residue question: "is this position part of an $\alpha$-helix?" on fixed-length fragments of amino-acid sequences. To ensure the setting remains transparent and reproducible, the project utilizes a two-state Hidden Markov Model (HMM) with fixed dynamics and emissions as the generative simulator. The main inference target is the position-wise posterior $p(S_{t}=1|X_{1:T})$, approximated using an amortized neural posterior with BayesFlow.

## Data and Generative Model
Synthetic protein sequence fragments of length 50 are generated over the standard 20-letter amino-acid alphabet.
*   **Latent State:** The sequence is represented by a binary latent state, $S_t \in \{0 = \text{other}, 1 = \alpha\text{-helix}\}$. The initialization is deterministic at $S_1 = 0$.
*   **State Dynamics:** The HMM is designed to favor persistence within both states. From the "other" state, the probability of remaining is 0.95, and the probability of entering "$\alpha$-helix" is 0.05. From the "$\alpha$-helix" state, the sequence remains with a probability of 0.90 and transitions to "other" with a probability of 0.10.
*   **Emissions:** Conditioned on the current state, the subsequent amino acid is drawn from a categorical distribution based on fixed empirical probabilities. Helix-prone residues (e.g., A, L, E) receive higher probability mass in the $\alpha$ state, while residues like G are more likely in the "other" state.

## Methodology

### Statistical Model (Simulator)
The HMM defines a latent sequence and corresponding observations, with the joint distribution calculated as:
$$p(S_{1:T},X_{1:T})=Pr(S_{1})\prod_{t=2}^{T}Pr(S_{t}|S_{t-1})\prod_{t=1}^{T}Pr(X_{t}|S_{t})$$

### Exact Inference (Baseline)
An exact HMM inference baseline is established using the Forward-Backward (FB) algorithm via the `hmmlearn` Python package. The smoothed posterior probability is computed as:
$$\gamma_{t}(s)=Pr(S_{t}=s|X_{1:T})=\frac{\alpha_{t}(s)\beta_{t}(s)}{\sum_{u\in\{0,1\}}\alpha_{t}(u)\beta_{t}(u)}$$

### Approximator (Amortized Inference)
While the HMM allows for exact inference, it is restricted to fixed parameters. Therefore, an amortized posterior map is learned utilizing **BayesFlow**.
*   **Input Representation:** Each sequence is converted into a one-hot matrix $X\in\{0,1\}^{T\times20}$.
*   **Summary Network:** The input matrix is flattened to preserve positional order and passed through an MLP with hidden sizes of 256, 128, and 64, utilizing GELU activations. This design replaced an earlier TimeSeries Network (which used mean pooling) to prevent the loss of order information critical for per-position inference.
*   **Posterior Approximation:** A coupling-flow network outputs a distribution over logits ($\eta_{1:T}$), which are then passed through a sigmoid function to yield per-position probabilities.

## Training Configuration
*   **Data Splits:** The model is trained offline using 6000 training sequences and 600 validation sequences.
*   **Hyperparameters:** Training runs for 15 epochs with a batch size of 32, 100 steps per epoch, and an initial learning rate of $10^{-3}$.
*   **Objective:** The objective minimizes the simulation-based conditional density learning loss, maximizing the conditional likelihood of simulated per-position logits.

## Diagnostics and Results

### Training Convergence
The training loss decreased steadily, but validation loss flattened and became noisy after 7-8 epochs. Attempts to force the training and validation lines to converge entirely resulted in unstable optimization, NaNs, and poor Simulation-Based Calibration (SBC) diagnostics.

### Calibration and Recovery
*   **SBC (ECDF):** The empirical CDF curves bent away from the diagonal across many positions, indicating miscalibration. This was largely driven by class imbalance, a tight summary bottleneck, and the flow density struggling with extreme logit target spikes.
*   **Recovery:** The posterior summaries clustered near the negative extreme, revealing a bias where the amortizer frequently predicted the majority class ("other") regardless of sequence context.

### Evaluation on Real Data (Human Insulin)
The models were evaluated against DSSP annotations for the human insulin sequence (PDB ID: 1A7F). The exact FB baseline proved highly capable, whereas the amortizer remained conservative with wide intervals.
*   **Chain A:** BayesFlow ROC-AUC = 0.75; FB ROC-AUC = 0.97.
*   **Chain B:** BayesFlow ROC-AUC = 0.44; FB ROC-AUC = 0.98.

## Discussion & Future Work
The project highlights that while amortized inference offers speed and flexibility, matching an exact FB baseline under a perfectly specified HMM is challenging. Future improvements to close this gap include:
*   Using a Bernoulli head to predict probabilities directly, avoiding continuous flow issues with discrete targets.
*   Replacing the MLP bottleneck with a time-aware summary network like a 1D CNN or a Transformer with positional encodings.
*   Implementing techniques to rebalance classes or use a weighted loss.
*   Training for more epochs combined with post hoc temperature calibration.

## Author
**Syed Hamza Afzal Ashraf**  
*Project completed for "Simulation Based Inference" (Group 2), TU Dortmund University*