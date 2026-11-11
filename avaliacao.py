import numpy as np
from sklearn import metrics
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt  # <-- NOVA IMPORTAÇÃO AQUI

"""
Biblioteca de Métricas para Detecção de Forgery em Multimídia.
Baseado nos protocolos de Castro et al. (2026).

Convenções:
- y_true: 0 para Autêntico (H0), 1 para Manipulado (H1)
- y_prob: Probabilidade/Score do evento ser H1 (Manipulado)
"""

def calculate_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calcula a Área sob a Curva ROC (AUC).
    Ref: Seção 3.5.1, Item (c).
    """
    return metrics.roc_auc_score(y_true, y_prob)

def calculate_eer(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calcula o Equal Error Rate (EER).
    Ref: Seção 3.5.1, Item (d) e Eq. (8).
    
    O EER ocorre quando P_FA (Falso Alarme) == P_M (Perda/Miss).
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[eer_idx]

def calculate_pd_at_fixed_pfa(y_true: np.ndarray, y_prob: np.ndarray, target_pfa: float = 0.01) -> float:
    """
    Calcula a Taxa de Detecção (Pd) para uma taxa fixa de Falso Alarme (Pfa).
    Ref: Seção 3.5.1, Item (e).
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    interp_func = interp1d(fpr, tpr, kind='linear')
    target_pfa = np.clip(target_pfa, 0, 1)
    return float(interp_func(target_pfa))

def get_det_curve_values(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Retorna os valores para plotagem da Curva DET (Detection Error Trade-off).
    Ref: Seção 3.5.1, Item (b).
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    p_m = 1 - tpr 
    p_fa = fpr    
    return p_fa, p_m

def get_roc_curve_values(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Retorna os valores para plotagem da Curva ROC.
    Ref: Seção 3.5.1, Item (a).
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    return fpr, tpr, thresholds

def get_threshold_at_eer(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Encontra o limiar exato onde ocorre o EER.
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    return thresholds[eer_idx]

def calculate_secondary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calcula Precision, Recall e F1-score em um ponto de operação fixo.
    Ref: Seção 3.5.2 .
    """
    y_pred = (y_prob >= threshold).astype(int)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "threshold_used": threshold
    }

def plot_evaluation_curves(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
    """
    Gera e plota as curvas ROC e DET lado a lado.
    
    Args:
        y_true: Array com os rótulos verdadeiros (0 ou 1).
        y_prob: Array com os scores/probabilidades previstos pelo modelo.
        save_path: Caminho opcional para salvar a imagem (ex: 'curvas_resultado.png'). 
                   Se None, exibe o gráfico na tela.
    """
    # Calcula valores e métricas
    fpr, tpr, _ = get_roc_curve_values(y_true, y_prob)
    p_fa, p_m = get_det_curve_values(y_true, y_prob)
    auc_val = calculate_auc(y_true, y_prob)
    eer_val = calculate_eer(y_true, y_prob)
    
    # Cria a figura com dois gráficos lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Gráfico 1: Curva ROC ---
    axs[0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_val:.4f})')
    axs[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--') # Linha de aleatoriedade
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('False Positive Rate (Falso Alarme)')
    axs[0].set_ylabel('True Positive Rate (Taxa de Detecção)')
    axs[0].set_title('Curva ROC')
    axs[0].legend(loc="lower right")
    axs[0].grid(alpha=0.3)
    
    # --- Gráfico 2: Curva DET ---
    # Geralmente plotada em escala logarítmica para evidenciar as taxas de erro
    axs[1].plot(p_fa, p_m, color='red', lw=2, label=f'DET Curve (EER = {eer_val:.4f})')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('False Alarm Probability (P_FA)')
    axs[1].set_ylabel('Miss Probability (P_M)')
    axs[1].set_title('Curva DET')
    axs[1].legend(loc="upper right")
    axs[1].grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    
    # Salva ou exibe
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de avaliação salvo em: {save_path}")
    else:
        plt.show()
        
    plt.close()
