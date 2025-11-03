"""
Interactive Visualization Module for Multimodal Review Analyzer.

This module provides comprehensive visualization capabilities using Plotly
for creating interactive charts, comparison visualizations, and HTML dashboards
for model performance analysis.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json


def plot_rating_distribution(df, rating_column="rating", key="rating_dist"):
    """
    Plot rating distribution histogram.
    
    Creates an interactive histogram showing the distribution of ratings
    in the dataset using Plotly.
    
    Args:
        df (pandas.DataFrame): DataFrame containing rating data
        rating_column (str): Name of the rating column (default: "rating")
        key (str): Unique key for Streamlit component caching
    """
    fig = px.histogram(df, x=rating_column, nbins=5, title="Rating Distribution")
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_sentiment_chart(sentiment_df, key="sentiment_chart"):
    """
    Plot sentiment distribution bar chart.
    
    Creates an interactive bar chart showing the distribution of sentiment
    categories in the data.
    
    Args:
        sentiment_df (pandas.DataFrame): DataFrame with 'sentiment' and 'count' columns
        key (str): Unique key for Streamlit component caching
    """
    fig = px.bar(sentiment_df, x="sentiment", y="count", color="sentiment", title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True, key=key)

def show_dashboard(stats_dict):
    """
    Display a dashboard of metrics using Streamlit metric components.
    
    Args:
        stats_dict (dict): Dictionary mapping metric names to values
    """
    for k, v in stats_dict.items():
        st.metric(label=k, value=v)

def create_model_comparison_chart(results: Dict[str, Dict[str, float]], task_type: str = "classification") -> go.Figure:
    """
    Create comprehensive model comparison chart.
    
    Generates a multi-subplot chart comparing different models across multiple
    performance metrics using Plotly.
    
    Args:
        results (dict): Dictionary mapping model names to their metric dictionaries
        task_type (str): Type of task - 'classification' or 'regression'
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure with comparison charts
    """
    
    models = list(results.keys())
    
    if task_type == "classification":
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    else:
        metrics = ['mae', 'mse', 'rmse']
        metric_labels = ['MAE', 'MSE', 'RMSE']
    
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=metric_labels,
        specs=[[{"secondary_y": False} for _ in range(len(metrics))]]
    )
    
    colors = px.colors.qualitative.Set3[:len(models)]
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric_labels[i],
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Add value labels on bars
        for j, v in enumerate(values):
            fig.add_annotation(
                x=models[j], y=v,
                text=f"{v:.3f}",
                showarrow=False,
                font=dict(size=10),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title=f"Model Performance Comparison - {task_type.title()}",
        height=400,
        showlegend=False
    )
    
    return fig

def create_fusion_technique_comparison(models_results: Dict[str, Dict[str, Dict[str, float]]]) -> go.Figure:
    """
    Create comparison chart for different fusion techniques.
    
    Generates a multi-panel visualization comparing Early, Late, Hybrid, and
    Cross-Modal Transformer fusion strategies across classification and regression tasks.
    
    Args:
        models_results (dict): Nested dictionary with structure:
            {task_type: {model_name: {metric: value}}}
            
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure with fusion comparison
    """
    
    fusion_models = ['EarlyFusion', 'LateFusion', 'HybridFusion', 'CrossModalTransformer']
    tasks = ['classification', 'regression']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{task.title()} - {metric.title()}" for task in tasks for metric in ['Accuracy/F1', 'MAE/RMSE']],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Classification metrics
    for task in tasks:
        if task == 'classification':
            metrics = ['accuracy', 'f1_score']
            row = 1
        else:
            metrics = ['mae', 'rmse']
            row = 2
        
        for col, metric in enumerate(metrics, 1):
            values = []
            for model in fusion_models:
                if model in models_results.get(task, {}):
                    values.append(models_results[task][model].get(metric, 0))
                else:
                    values.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=fusion_models,
                    y=values,
                    name=f"{task}_{metric}",
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title="Fusion Technique Performance Comparison",
        height=600,
        showlegend=False
    )
    
    return fig

def create_training_progress_chart(history: Dict[str, List[float]], task_type: str) -> go.Figure:
    """
    Create training progress visualization.
    
    Generates plots showing training and validation loss/accuracy over epochs.
    
    Args:
        history (dict): Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' keys
        task_type (str): Type of task - 'classification' or 'regression'
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure with training curves
    """
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Loss', 'Accuracy' if task_type == 'classification' else 'RMSE'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], mode='lines', name='Train Loss', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], mode='lines', name='Validation Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # Accuracy/RMSE plot
    if task_type == 'classification':
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_acc'], mode='lines', name='Train Accuracy', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_acc'], mode='lines', name='Validation Accuracy', line=dict(color='orange')),
            row=1, col=2
        )
    else:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_rmse'], mode='lines', name='Train RMSE', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_rmse'], mode='lines', name='Validation RMSE', line=dict(color='orange')),
            row=1, col=2
        )
    
    fig.update_layout(
        title="Training Progress",
        height=400,
        showlegend=True
    )
    
    return fig

def create_cross_modal_analysis_chart(text_features: Dict, numerical_features: Dict) -> go.Figure:
    """
    Create cross-modal feature analysis visualization.
    
    Generates a multi-panel chart showing text features, numerical features,
    cross-modal interactions, and feature importance.
    
    Args:
        text_features (dict): Dictionary of text-based features
        numerical_features (dict): Dictionary of numerical features
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure with feature analysis
    """
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Text Features', 'Numerical Features', 'Cross-Modal Interactions', 'Feature Importance'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Text features
    text_keys = list(text_features.keys())[:10]  # Limit to first 10
    text_values = [text_features[key] for key in text_keys]
    fig.add_trace(
        go.Bar(x=text_keys, y=text_values, name='Text Features', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Numerical features
    num_keys = list(numerical_features.keys())
    num_values = [numerical_features[key] for key in num_keys]
    fig.add_trace(
        go.Bar(x=num_keys, y=num_values, name='Numerical Features', marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Cross-modal interactions (scatter plot)
    if 'polarity' in text_features and 'rating' in numerical_features:
        fig.add_trace(
            go.Scatter(
                x=[text_features['polarity']],
                y=[numerical_features['rating']],
                mode='markers',
                name='Sentiment vs Rating',
                marker=dict(size=15, color='green')
            ),
            row=2, col=1
        )
    
    # Feature importance (mock data for demonstration)
    importance_data = {
        'Feature': ['Text Sentiment', 'Rating', 'Review Length', 'Helpful Votes', 'Cross-Modal Fusion'],
        'Importance': [0.85, 0.78, 0.65, 0.52, 0.91]
    }
    fig.add_trace(
        go.Bar(x=importance_data['Feature'], y=importance_data['Importance'], 
               name='Feature Importance', marker_color='lightgreen'),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Cross-Modal Feature Analysis",
        height=600,
        showlegend=False
    )
    
    return fig

def create_performance_radar_chart(models_results: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Create radar chart for model performance comparison.
    
    Generates a polar (radar) chart showing multiple models' performance across
    different metrics simultaneously.
    
    Args:
        models_results (dict): Dictionary mapping model names to their metric dictionaries
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly radar chart
    """
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'mae', 'mse']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MAE', 'MSE']
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i, (model, results) in enumerate(models_results.items()):
        values = []
        for metric in metrics:
            value = results.get(metric, 0)
            # Normalize MAE and MSE (invert for radar chart)
            if metric in ['mae', 'mse']:
                value = 1 / (1 + value) if value > 0 else 0
            values.append(value)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels,
            fill='toself',
            name=model,
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Model Performance Radar Chart",
        height=500
    )
    
    return fig

def create_interactive_dashboard_html(results: Dict[str, Any]) -> str:
    """
    Generate interactive HTML dashboard.
    
    Creates a comprehensive HTML report with embedded Plotly charts and
    interactive visualizations for model performance analysis.
    
    Args:
        results (dict): Dictionary mapping model names to their metric dictionaries
        
    Returns:
        str: HTML string containing the complete interactive dashboard
    """
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multimodal Review Analyzer - Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 3px solid #667eea;
            }
            .header h1 {
                color: #667eea;
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .header p {
                color: #666;
                margin: 10px 0 0 0;
                font-size: 1.1em;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            .metric-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 10px 0;
            }
            .metric-label {
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .chart-container {
                margin: 30px 0;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            }
            .chart-title {
                text-align: center;
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.3em;
                font-weight: 500;
            }
            .fusion-comparison {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                margin: 30px 0;
                text-align: center;
            }
            .fusion-comparison h3 {
                margin: 0 0 20px 0;
                font-size: 1.5em;
            }
            .fusion-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .fusion-stat {
                background: rgba(255,255,255,0.1);
                padding: 15px;
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }
            .fusion-stat h4 {
                margin: 0 0 10px 0;
                font-size: 1.1em;
            }
            .fusion-stat .value {
                font-size: 1.5em;
                font-weight: bold;
                margin: 5px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Multimodal Review Analyzer</h1>
                <p>Advanced Fusion Techniques & Cross-Modal Analysis</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Best Model</div>
                    <div class="metric-value">{best_model}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{best_accuracy:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">{best_f1:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">{best_rmse:.3f}</div>
                </div>
            </div>
            
            <div class="fusion-comparison">
                <h3>Fusion Technique Performance</h3>
                <p>Comparative analysis of Early Fusion, Late Fusion, Hybrid Fusion, and Cross-Modal Transformer approaches</p>
                <div class="fusion-stats">
                    <div class="fusion-stat">
                        <h4>Early Fusion</h4>
                        <div class="value">{early_fusion_acc:.3f}</div>
                        <small>Accuracy</small>
                    </div>
                    <div class="fusion-stat">
                        <h4>Late Fusion</h4>
                        <div class="value">{late_fusion_acc:.3f}</div>
                        <small>Accuracy</small>
                    </div>
                    <div class="fusion-stat">
                        <h4>Hybrid Fusion</h4>
                        <div class="value">{hybrid_fusion_acc:.3f}</div>
                        <small>Accuracy</small>
                    </div>
                    <div class="fusion-stat">
                        <h4>Cross-Modal</h4>
                        <div class="value">{cross_modal_acc:.3f}</div>
                        <small>Accuracy</small>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Model Performance Comparison</div>
                <div id="model-comparison"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Fusion Technique Analysis</div>
                <div id="fusion-comparison"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Cross-Modal Feature Importance</div>
                <div id="feature-importance"></div>
            </div>
        </div>
        
        <script>
            // Model comparison chart
            const modelData = {model_comparison_data};
            Plotly.newPlot('model-comparison', modelData.data, modelData.layout);
            
            // Fusion comparison chart
            const fusionData = {fusion_comparison_data};
            Plotly.newPlot('fusion-comparison', fusionData.data, fusionData.layout);
            
            // Feature importance chart
            const featureData = {feature_importance_data};
            Plotly.newPlot('feature-importance', featureData.data, featureData.layout);
        </script>
    </body>
    </html>
    """
    
    # Extract data from results
    best_model = max(results.items(), key=lambda x: x[1].get('accuracy', 0))[0]
    best_accuracy = results.get(best_model, {}).get('accuracy', 0)
    best_f1 = results.get(best_model, {}).get('f1_score', 0)
    best_rmse = results.get(best_model, {}).get('rmse', 0)
    
    # Fusion technique accuracies
    early_fusion_acc = results.get('EarlyFusion', {}).get('accuracy', 0)
    late_fusion_acc = results.get('LateFusion', {}).get('accuracy', 0)
    hybrid_fusion_acc = results.get('HybridFusion', {}).get('accuracy', 0)
    cross_modal_acc = results.get('CrossModalTransformer', {}).get('accuracy', 0)
    
    # Mock chart data (in real implementation, this would come from actual results)
    model_comparison_data = {
        "data": [{
            "x": list(results.keys()),
            "y": [results[model].get('accuracy', 0) for model in results.keys()],
            "type": "bar",
            "marker": {"color": "#667eea"}
        }],
        "layout": {
            "title": "Model Accuracy Comparison",
            "xaxis": {"title": "Models"},
            "yaxis": {"title": "Accuracy"}
        }
    }
    
    fusion_comparison_data = {
        "data": [{
            "x": ["Early Fusion", "Late Fusion", "Hybrid Fusion", "Cross-Modal"],
            "y": [early_fusion_acc, late_fusion_acc, hybrid_fusion_acc, cross_modal_acc],
            "type": "bar",
            "marker": {"color": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]}
        }],
        "layout": {
            "title": "Fusion Technique Performance",
            "xaxis": {"title": "Fusion Techniques"},
            "yaxis": {"title": "Accuracy"}
        }
    }
    
    feature_importance_data = {
        "data": [{
            "x": ["Text Sentiment", "Rating", "Review Length", "Cross-Modal Fusion"],
            "y": [0.85, 0.78, 0.65, 0.91],
            "type": "bar",
            "marker": {"color": "#96CEB4"}
        }],
        "layout": {
            "title": "Feature Importance",
            "xaxis": {"title": "Features"},
            "yaxis": {"title": "Importance Score"}
        }
    }
    
    return html_template.format(
        best_model=best_model,
        best_accuracy=best_accuracy,
        best_f1=best_f1,
        best_rmse=best_rmse,
        early_fusion_acc=early_fusion_acc,
        late_fusion_acc=late_fusion_acc,
        hybrid_fusion_acc=hybrid_fusion_acc,
        cross_modal_acc=cross_modal_acc,
        model_comparison_data=json.dumps(model_comparison_data),
        fusion_comparison_data=json.dumps(fusion_comparison_data),
        feature_importance_data=json.dumps(feature_importance_data)
    )

def export_results_to_html(results: Dict[str, Any], filename: str = "multimodal_analysis_report.html"):
    """
    Export analysis results to interactive HTML report.
    
    Creates and saves an interactive HTML file with model performance visualizations.
    
    Args:
        results (dict): Dictionary mapping model names to their metric dictionaries
        filename (str): Output filename for the HTML report (default: "multimodal_analysis_report.html")
        
    Returns:
        str: Path to the saved HTML file
    """
    
    html_content = create_interactive_dashboard_html(results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename
