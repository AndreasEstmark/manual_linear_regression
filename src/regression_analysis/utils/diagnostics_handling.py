import numpy as np


def handle_high_vif_columns(X: np.ndarray, vifs: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """
    Handles multicollinearity by prompting the user to remove features with high VIF.

    Parameters:
        X: Design matrix
        vifs: Dict of feature index -> VIF
        threshold: VIF value to trigger removal suggestion

    Returns:
        Possibly reduced X (features removed if user agrees)
    """

    high_vif_indices = [i for i, vif in enumerate(vifs) if vif > threshold]

    if not high_vif_indices:
        return X


    print("\n⚠️ High VIF detected for the following features:")
    for i in high_vif_indices:
        print(f" x_{i} with VIF = {vifs[i]:.2f}")

    user_choice = input ("Would you like to drop these features? (y/n): ".strip().lower())

    if user_choice == "y":
        X = np.delete(X, high_vif_indices,axis =1)
        print(f"Dropped columns: {high_vif_indices}")
    else:
        print("Proceeding without dropping features.")

    return X 

                         
                        