from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_auc_roc_curve(y, y_proba):
    """
    Plots the AUC-ROC curve.

    Parameters:
    - y: array-like, true binary labels
    - y_proba: array-like, predicted probabilities for the positive class
    """
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    
    # Calculate the AUC score
    auc_score = roc_auc_score(y, y_proba)
    print("AUC Score:", auc_score)

    # Plot the ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()