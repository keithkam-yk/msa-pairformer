import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


def fit_seq_weight_mixture_model(
    seq_weights_a: np.array,
    n_components: int = 2,
    random_state: int = 42,
    max_iter: int = 5000,
    n_init: int = 100,
    tol: float = 1e-6,
    scaling_factor = 1e5,
    plot = True,
    figsize = (6, 4),
    return_gmm = False
):
    # Fit a mixture model to the sequence weights
    # Prepare the data
    X = seq_weights_a.reshape(-1, 1)
    # Initialize and fit GMM to scaled data
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, max_iter=max_iter, n_init=n_init, tol=tol)
    gmm.fit(X * scaling_factor)
    average_log_likelihood = gmm.score(X * scaling_factor)
    n_samples = X.shape[0]
    total_log_likelihood = average_log_likelihood * n_samples
    # Generate x-values in original scale
    x = np.linspace(seq_weights_a.min(), seq_weights_a.max(), 1000).reshape(-1, 1)
    # Compute PDFs in scaled space
    x_scaled = x * scaling_factor
    log_prob_scaled = gmm.score_samples(x_scaled)
    pdf_scaled = np.exp(log_prob_scaled)
    responsibilities = gmm.predict_proba(x_scaled)
    pdf_individual_scaled = responsibilities * pdf_scaled[:, np.newaxis]
    
    if plot:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot histogram of original data with density=False to show counts
        hist_counts, hist_bins, _ = ax.hist(seq_weights_a, bins=100, density=False, 
                                          alpha=0.5, color="grey", 
                                          label="Avg. sequence attention weights")
        
        # Calculate bin width for scaling PDFs to match count data
        bin_width = hist_bins[1] - hist_bins[0]
        
        # Scale factor to convert density to counts
        scale_to_counts = bin_width * len(seq_weights_a)
        
        # Plot the PDFs scaled to match counts
        for component_idx in range(n_components):
            # Transform PDFs to count scale
            pdf_component = pdf_individual_scaled[:, component_idx] * scaling_factor * scale_to_counts
            ax.plot(x, pdf_component, '--', color=f'C{component_idx}', label=f"Component {component_idx}")
        
        # Set labels and title
        ax.set_xlabel("Sequence weight", size=12)
        ax.set_ylabel("Count", size=12)  # Changed from "Density" to "Count"
        ax.set_title("OmpR sequence weight distribution", size=12)  # Updated title to match your image
        
        # Add vertical line at x = 1/num_seqs
        ax.axvline(x=1 / seq_weights_a.shape[0], color='grey', linestyle='--', label="Uniform weighting")
        ax.legend(fontsize=10)
        
        return gmm.means_ / scaling_factor, gmm.covariances_ / scaling_factor**2, total_log_likelihood, f, ax
    
    return gmm.means_ / scaling_factor, gmm.covariances_ / scaling_factor**2, total_log_likelihood
