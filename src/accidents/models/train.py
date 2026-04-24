from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score,
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score, 
    balanced_accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def train_and_eval_classification(models, X_train, y_train, X_val, y_val, strategy='', k=None, verbose=True, ):
    """
    Train and evaluate multiple classification models on validation data.
    
    Parameters:
        models (dict): Dictionary of model names to sklearn estimators.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.
        strategy (str, optional): The strategy used for training.
        k (int, optional): The number of PCA components if applicable.
        verbose (bool, optional): Whether to print detailed results and show confusion matrix plots.
    
    Returns:
        trained_models (dict): Dictionary of trained models.
        results (dict): Dictionary of model names to validation accuracy scores.
    """
    
    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_val_pred)
        trained_models[name] = model
        if verbose:
            print("="*70)
            print(f'{name}: validation accuracy = {score:.4f}')
            print("Classification report:")
            print(classification_report(y_val, y_val_pred))
            
        f1_per_class = f1_score(
            y_val,
            y_val_pred,
            labels=['Grave', 'Leger', 'Materiel'],
            average=None,
            zero_division=0
        )

        results.append({
            'Strategy': strategy,
            'PCA-k': k,
            'Model': name,
            'Accuracy': accuracy_score(y_val, y_val_pred),
            'Balanced Acc': balanced_accuracy_score(y_val, y_val_pred),
            'F1 macro': f1_score(y_val, y_val_pred, average='macro', zero_division=0),
            'F1 Grave': f1_per_class[0],
        })
        
        # plot confusion matrix
        if verbose:
            sns.heatmap(confusion_matrix(y_val, y_val_pred), annot=True, fmt='d', cmap='Blues')
            plt.title(f"{name} Confusion Matrix")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

    if verbose:
        results_df = pd.DataFrame(results)
        print('Résumé des performances :')
        print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    return trained_models, results

def train_and_eval_clustering(models, X_train, X_val, y_val, verbose=True):
    """
    Train and evaluate multiple clustering models on validation data.
    
    Parameters:
        models (dict): Dictionary of model names to sklearn clusterers.
        X_train (array-like): Training features (used for fitting).
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels (true clusters for evaluation).
    
    Returns:
        results (dict): Dictionary of model names to evaluation metrics dict.
    """
    
    results = []
    for name, model in models.items():
        model.fit(X_train)
        if hasattr(model, 'predict'):
            labels_val = model.predict(X_val)
        else:
            labels_val = model.fit_predict(X_val)

        ari = adjusted_rand_score(y_val, labels_val)
        ami = adjusted_mutual_info_score(y_val, labels_val)
        hom = homogeneity_score(y_val, labels_val)
        v = v_measure_score(y_val, labels_val)
        sil = silhouette_score(X_val, labels_val)

        results.append({
            'Model': name,
            'ARI': ari,
            'AMI': ami,
            'Homogeneity': hom,
            'Silhouette': sil,
        })

        
        if verbose:
            print(f'{name}:')
            print(f'  ARI: {ari:.4f}')
            print(f'  AMI: {ami:.4f}')
            print(f'  Homogeneity: {hom:.4f}')
            print(f'  Silhouette: {sil:.4f}')

    if verbose:
        results_df = pd.DataFrame(results)
        print('Résumé des performances :')
        print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        
    return results

def train_and_eval(models, X_train, y_train, X_val, y_val, task_type='classification', strategy='', k=None, verbose=True):
    """
    Train and evaluate models based on task type.
    
    Parameters:
        models (dict): Dictionary of model names to estimators/clusterers.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.
        task_type (str): 'classification' or 'clustering'.
        strategy (str, optional): The strategy used for training (for classification).
        k (int, optional): The number of PCA components if applicable (for classification).
        verbose (bool): For classification, whether to print details and show plots.
    
    Returns:
        For classification: tuple of (trained_models, results)
        For clustering: results dict
    """
    if task_type == 'classification':
        return train_and_eval_classification(models, X_train, y_train, X_val, y_val, strategy, k, verbose)
    elif task_type == 'clustering':
        return train_and_eval_clustering(models, X_train, X_val, y_val, verbose)
    else:
        raise ValueError(f"Unknown task type: {task_type}")