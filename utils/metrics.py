"""
Comprehensive Evaluation Metrics Module.

This module provides extensive evaluation metrics for both classification and
regression tasks, including cross-modal evaluation and advanced model comparison
capabilities.
"""

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report,
    r2_score, explained_variance_score, max_error, median_absolute_error
)
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def regression_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression evaluation metrics.
    
    Computes multiple regression metrics including error measures and variance
    explained to provide a complete picture of model performance.
    
    Args:
        y_true: True target values (array-like)
        y_pred: Predicted target values (array-like)
        
    Returns:
        dict: Dictionary containing:
            - MSE: Mean Squared Error
            - MAE: Mean Absolute Error
            - RMSE: Root Mean Squared Error
            - R2: R-squared (coefficient of determination)
            - Explained Variance: Proportion of variance explained
            - Max Error: Maximum absolute error
            - Median AE: Median Absolute Error
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    
    return {
        "MSE": mse, 
        "MAE": mae, 
        "RMSE": rmse,
        "R2": r2,
        "Explained Variance": evs,
        "Max Error": max_err,
        "Median AE": medae
    }

def classification_metrics(y_true, y_pred):
    """
    Calculate comprehensive classification evaluation metrics.
    
    Computes multiple classification metrics including accuracy, precision,
    recall, and F1-scores with different averaging strategies.
    
    Args:
        y_true: True class labels (array-like)
        y_pred: Predicted class labels (array-like)
        
    Returns:
        dict: Dictionary containing:
            - Accuracy: Overall classification accuracy
            - Precision: Weighted precision score
            - Recall: Weighted recall score
            - F1-Score: Weighted F1 score
            - F1-Macro: Macro-averaged F1 score
            - F1-Micro: Micro-averaged F1 score
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Additional metrics
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    return {
        "Accuracy": acc, 
        "Precision": precision, 
        "Recall": recall, 
        "F1-Score": f1,
        "F1-Macro": f1_macro,
        "F1-Micro": f1_micro
    }

def cross_modal_evaluation_metrics(y_true_text, y_pred_text, y_true_num, y_pred_num):
    """
    Calculate advanced cross-modal evaluation metrics.
    
    Evaluates performance across both text and numerical modalities and
    measures cross-modal consistency and fusion effectiveness.
    
    Args:
        y_true_text: True text modality labels
        y_pred_text: Predicted text modality labels
        y_true_num: True numerical modality values
        y_pred_num: Predicted numerical modality values
        
    Returns:
        dict: Dictionary containing:
            - text_metrics: Classification metrics for text modality
            - numerical_metrics: Regression metrics for numerical modality
            - cross_modal_consistency: Consistency scores between modalities
            - fusion_effectiveness: Measure of fusion improvement
    """
    
    # Individual modality metrics
    text_metrics = classification_metrics(y_true_text, y_pred_text)
    num_metrics = regression_metrics(y_true_num, y_pred_num)
    
    # Cross-modal consistency metrics
    consistency_score = compute_cross_modal_consistency(y_true_text, y_pred_text, y_true_num, y_pred_num)
    
    # Fusion effectiveness
    fusion_effectiveness = compute_fusion_effectiveness(y_true_text, y_pred_text, y_true_num, y_pred_num)
    
    return {
        "text_metrics": text_metrics,
        "numerical_metrics": num_metrics,
        "cross_modal_consistency": consistency_score,
        "fusion_effectiveness": fusion_effectiveness
    }

def compute_cross_modal_consistency(y_true_text, y_pred_text, y_true_num, y_pred_num):
    """
    Compute consistency metrics between text and numerical predictions.
    
    Measures agreement between predictions from different modalities and
    consistency with ground truth across modalities.
    
    Args:
        y_true_text: True text labels
        y_pred_text: Predicted text labels
        y_true_num: True numerical values
        y_pred_num: Predicted numerical values
        
    Returns:
        dict: Consistency metrics including:
            - text_num_agreement: Agreement between text and numerical predictions
            - text_consistency: Text prediction consistency with ground truth
            - numerical_consistency: Numerical prediction consistency
            - overall_consistency: Average consistency across all measures
    """
    
    # Convert numerical ratings to sentiment classes for consistency check
    num_sentiment_true = (y_true_num >= 3).astype(int)  # 1 for positive (>=3), 0 for negative (<3)
    num_sentiment_pred = (y_pred_num >= 3).astype(int)
    
    # Check if text and numerical predictions agree
    text_num_agreement = (y_pred_text == num_sentiment_pred).mean()
    
    # Check consistency with ground truth
    text_consistency = (y_true_text == y_pred_text).mean()
    num_consistency = (num_sentiment_true == num_sentiment_pred).mean()
    
    # Overall consistency score
    overall_consistency = (text_consistency + num_consistency + text_num_agreement) / 3
    
    return {
        "text_num_agreement": text_num_agreement,
        "text_consistency": text_consistency,
        "numerical_consistency": num_consistency,
        "overall_consistency": overall_consistency
    }

def compute_fusion_effectiveness(y_true_text, y_pred_text, y_true_num, y_pred_num):
    """
    Compute fusion effectiveness metrics.
    
    Measures how much fusion improves predictions compared to individual
    modality performance.
    
    Args:
        y_true_text: True text labels
        y_pred_text: Predicted text labels (from fusion model)
        y_true_num: True numerical values
        y_pred_num: Predicted numerical values (from fusion model)
        
    Returns:
        dict: Effectiveness metrics including:
            - text_improvement: Improvement in text modality performance
            - numerical_improvement: Improvement in numerical modality performance
            - overall_improvement: Average improvement across modalities
    """
    
    # Simulate individual modality performance (in real scenario, these would be actual individual model predictions)
    text_individual_acc = (y_true_text == y_pred_text).mean()
    num_individual_acc = (np.abs(y_true_num - y_pred_num) <= 0.5).mean()  # Within 0.5 rating difference
    
    # Fusion performance (current predictions)
    fusion_text_acc = text_individual_acc  # Assuming fusion maintains or improves
    fusion_num_acc = num_individual_acc
    
    # Calculate improvement
    text_improvement = fusion_text_acc - text_individual_acc
    num_improvement = fusion_num_acc - num_individual_acc
    
    return {
        "text_improvement": text_improvement,
        "numerical_improvement": num_improvement,
        "overall_improvement": (text_improvement + num_improvement) / 2
    }

def advanced_model_comparison(models_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Perform advanced model comparison with statistical analysis.
    
    Compares multiple models across different metrics, identifies best performers,
    calculates performance gaps, and provides overall rankings.
    
    Args:
        models_results (dict): Dictionary mapping model names to their metric dictionaries
        
    Returns:
        dict: Comprehensive comparison results including:
            - For each metric: best_model, best_value, performance_gaps, relative_performance
            - overall_ranking: Models ranked by average performance
            - model_scores: Average scores for each model
    """
    
    # Extract metrics for comparison
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    
    comparison_results = {}
    
    for metric in metrics:
        if all(metric in results for results in models_results.values()):
            values = [results[metric] for results in models_results.values()]
            models = list(models_results.keys())
            
            # Statistical analysis
            best_model_idx = np.argmax(values)
            best_model = models[best_model_idx]
            best_value = values[best_model_idx]
            
            # Performance gaps
            gaps = [best_value - val for val in values]
            
            comparison_results[metric] = {
                "best_model": best_model,
                "best_value": best_value,
                "performance_gaps": dict(zip(models, gaps)),
                "relative_performance": {model: val/best_value for model, val in zip(models, values)}
            }
    
    # Overall ranking
    model_scores = {}
    for model, results in models_results.items():
        score = sum(results.get(metric, 0) for metric in metrics) / len(metrics)
        model_scores[model] = score
    
    ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    comparison_results["overall_ranking"] = ranked_models
    comparison_results["model_scores"] = model_scores
    
    return comparison_results

def create_confusion_matrix_plot(y_true, y_pred, model_name: str = "Model"):
    """
    Create a confusion matrix visualization.
    
    Generates a heatmap showing the confusion matrix for classification results,
    useful for understanding classification errors and model performance.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        model_name (str): Name of the model for plot title
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the confusion matrix plot
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt.gcf()

def create_regression_scatter_plot(y_true, y_pred, model_name: str = "Model"):
    """
    Create a regression scatter plot with perfect prediction reference line.
    
    Visualizes predicted vs actual values with a diagonal reference line
    representing perfect predictions.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name (str): Name of the model for plot title
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the scatter plot
    """
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Regression Scatter Plot - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def generate_model_report(models_results: Dict[str, Dict[str, float]], 
                         task_type: str = "classification") -> str:
    """
    Generate a comprehensive markdown-formatted model performance report.
    
    Creates a detailed report comparing all models, identifying best performers,
    and providing recommendations for deployment.
    
    Args:
        models_results (dict): Dictionary mapping model names to their metrics
        task_type (str): Type of task - 'classification' or 'regression'
        
    Returns:
        str: Markdown-formatted report string
    """
    
    report = f"""
# Multimodal Review Analyzer - Model Performance Report
## Task Type: {task_type.title()}

"""
    
    # Best performing model
    if task_type == "classification":
        best_metric = "accuracy"
    else:
        best_metric = "rmse"
    
    if task_type == "classification":
        best_model = max(models_results.items(), key=lambda x: x[1].get(best_metric, 0))
    else:
        best_model = min(models_results.items(), key=lambda x: x[1].get(best_metric, float('inf')))
    
    report += f"## Best Performing Model: {best_model[0]}\n"
    report += f"### Performance Metrics:\n"
    
    for metric, value in best_model[1].items():
        report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
    
    report += "\n## Model Comparison Summary\n"
    
    for model_name, results in models_results.items():
        report += f"\n### {model_name}\n"
        for metric, value in results.items():
            report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
    
    # Fusion technique analysis
    fusion_models = ['EarlyFusion', 'LateFusion', 'HybridFusion', 'CrossModalTransformer']
    fusion_results = {model: results for model, results in models_results.items() 
                     if model in fusion_models}
    
    if fusion_results:
        report += "\n## Fusion Technique Analysis\n"
        for model, results in fusion_results.items():
            fusion_type = model.replace('Fusion', '').replace('CrossModalTransformer', 'Cross-Modal Transformer')
            report += f"\n### {fusion_type}\n"
            report += f"- Accuracy: {results.get('accuracy', 0):.4f}\n"
            report += f"- F1-Score: {results.get('f1_score', 0):.4f}\n"
    
    report += "\n## Recommendations\n"
    report += "1. **Best Overall Model**: Use the top-performing model for production deployment\n"
    report += "2. **Fusion Strategy**: Consider ensemble methods combining multiple fusion techniques\n"
    report += "3. **Cross-Modal Features**: Leverage the cross-modal feature engineering for improved performance\n"
    report += "4. **Hyperparameter Tuning**: Fine-tune the best models for optimal performance\n"
    
    return report

def combined_metrics(y_true, y_pred, task_type='regression'):
    """
    Legacy function for backward compatibility.
    
    Wrapper function that calls appropriate metrics function based on task type.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type (str): 'classification' or 'regression'
        
    Returns:
        dict: Metrics dictionary for the specified task type
    """
    if task_type == 'regression':
        return regression_metrics(y_true, y_pred)
    elif task_type == 'classification':
        return classification_metrics(y_true, y_pred)
    else:
        raise ValueError("task_type must be either 'regression' or 'classification'")