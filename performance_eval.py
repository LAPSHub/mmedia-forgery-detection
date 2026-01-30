import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# to make it possible to use the functions to calculate the auc and trace roc 
# curves, we should take the output of softmax layer (?)

def auc(y_real: dict, y_prob: dict) -> float:
    """Calculate the area under the receiver operating characteristic (ROC) 
    curve - AUC for a face forgery detector.

    Parameters 
    ----------
    y_real : dict
        Ground-truth data. Dictionary with true class frame by frame.The names
        of the video files are the keys of the dictionary. 
    y_prob : dict
        Probabilities of the fake class given by the classifier frame by frame.
        The names of the video files are the keys of the dictionary.

    Returns
    -------
    float
        The value of the area under the receiver operating characteristic (ROC)
        curve.
        
    """
    u_real = unify( y_real )
    u_prob = unify( y_prob )
    score = metrics.roc_auc_score(u_real, u_prob)
    return score

def roc(y_real: dict, y_prob: dict):
    """Plot the receiver operating characteristic (ROC) curve - AUC for a 
    face forgery detector.

    Parameters 
    ----------
    y_real : dict
        Ground-truth data. Dictionary with true class frame by frame.The names
        of the video files are the keys of the dictionary. 
    y_prob : dict
        Probabilities of the fake class given by the classifier frame by frame.
        The names of the video files are the keys of the dictionary.

    """
    u_real = unify( y_real )
    u_prob = unify( y_prob )
    metrics.RocCurveDisplay.from_predictions( u_real, u_prob )
    plt.grid(True)
    plt.show()
    return None

def acc(y_real: dict, y_pred: dict) -> float:
    """Calculate the accuracy for a face forgery detector.

    Parameters 
    ----------
    y_real : dict
        Ground-truth data. Dictionary with true class frame by frame.The names
        of the video files are the keys of the dictionary. 
    y_pred : dict
        Predictions given by the classifier frame by frame. The names of the 
        video files are the keys of the dictionary.

    Returns
    -------
    float
        The value of the accuracy.

    """
    u_real = unify( y_real )
    u_pred = unify( y_pred )
    score = metrics.accuracy_score(u_real, u_pred)
    return score

def f1(y_real: dict, y_pred: dict) -> float:
    """Calculate the F1-score (harmonic mean of the precision and recall) for
    a face forgery detector.

    Parameters 
    ----------
    y_real : dict
        Ground-truth data. Dictionary with true class frame by frame.The names
        of the video files are the keys of the dictionary. 
    y_pred : dict
        Predictions given by the classifier frame by frame. The names of the 
        video files are the keys of the dictionary.

    Returns
    -------
    float
        The value of the F1-score.
   
    """
    u_real = unify( y_real )
    u_pred = unify( y_pred )
    score = metrics.f1_score(u_real, u_pred)
    return score

def eer(y_real: dict, y_prob: dict) -> float:
    """Calculate the equal error rate (EER) for a face forgery detector.

    Parameters 
    ----------
    y_real : dict
        Ground-truth data. Dictionary with true class frame by frame.The names
        of the video files are the keys of the dictionary. 
    y_prob : dict
        Probabilities of the fake class given by the classifier frame by frame.
        The names of the video files are the keys of the dictionary.

    Returns
    -------
    score : float
        The value of the EER.
    thres : float
        Threshold
    
    """
    u_real = unify( y_real )
    u_prob = unify( y_prob )
        
    fpr, tpr, thresholds = metrics.roc_curve(u_real, u_prob, pos_label=1)

    #fnr = 1-tpr
    #abs_diffs = np.abs(fpr - fnr)
    #min_index = np.argmin(abs_diffs)
    #eer1 = np.mean((fpr[min_index], fnr[min_index]))
    
    score = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thres = interp1d(fpr, thresholds)(score)
    return score, thres

def unify( dataset: dict ) -> list:
    """Receive a dataset dictionary and return a single list whose elements
    represents classes or probabilities of frames. 

    Parameters
    ----------
    dataset : dict
        Video dataset classes or probabilities. Dictionary keys are video file
        names; associated with each key, there is a list with information 
        (classes or probabilities) in a frame by frame basis. 

    Returns
    -------
    list
        Information (classes or probabilities) from each video frame of the 
        datase.

    """
    output = []
    for i in dataset:
        output.extend(dataset[i])
    return output

def prob2class(prob: dict, threshold: float) -> dict:
    """Convert probabilities to classes by using a threshold.

    Parameters
    ----------
    prob: dict
        Dictionary with video file names (keys) and output probabilites, given
        by a detector (at frame or video-level).
    threshold: float
        A number from which the probabilities are converted to classes
        (binary classification).

    Returns
    -------
    dict
        Dictionary with estimated classes.

    """
    out = {}
    for i in prob:
        out[i] = []
        for j in prob[i]:
            if j > threshold:
                out[i].append(1)
            else:
                out[i].append(0)
    return out

def frame2video(y: dict, option: str) -> dict:
    """Agregate information from all frames of a video into a single value.

    Parameters
    ----------
    y: dict
        Dictionary with information at frame-level. Video file names are
        the keys of the dictionary.
    option: str
        'probability': the aggregated value is the average probability of all
        frames.
        'ground_truth': the aggregated value corresponds to the more frequent
        class.

    Returns
    -------
    dict
        Dictionary with aggregated information for each video.

    """
    out = {}
    for i in y:
        out[i] = []
        if option == 'probability':
            out[i].append(np.mean(y[i]))
        elif option == 'ground_truth':
            if (2*np.sum(y[i])) > len(y[i]):
                out[i].append(1)
            else:
                out[i].append(0)
    return out

    

if __name__ == '__main__':
    #import sys 

    # Ground truth at frame-level
    y_gdt_frm = {'video 01': [ 0, 0, 0, 0, 1 ],
                 'video 02': [ 0, 0, 0, 1, 1 ],
                 'video 03': [ 0, 0, 1, 1, 1 ],
                 'video 04': [ 0, 1, 1, 1, 1 ],
                 'video 05': [ 1, 1, 1, 1, 1 ],
                 'video 06': [ 1, 1, 1, 1, 0 ],
                 'video 07': [ 1, 1, 1, 0, 0 ],
                 'video 08': [ 1, 1, 0, 0, 0 ],
                 'video 09': [ 1, 0, 0, 0, 0 ]
                 }

    # Probabilities of being fake at frame-level
    y_prb_frm = {'video 01': [ 0.10, 0.40, 0.20, 0.45, 0.55 ],
                 'video 02': [ 0.55, 0.45, 0.35, 0.65, 0.60 ],
                 'video 03': [ 0.25, 0.45, 0.45, 0.95, 0.90 ],
                 'video 04': [ 0.51, 0.65, 0.35, 0.65, 0.49 ],
                 'video 05': [ 0.15, 0.65, 0.51, 0.65, 0.60 ],
                 'video 06': [ 0.55, 0.45, 0.75, 0.65, 0.20 ],
                 'video 07': [ 0.55, 0.45, 0.75, 0.55, 0.20 ],
                 'video 08': [ 0.55, 0.45, 0.55, 0.55, 0.20 ],
                 'video 09': [ 0.85, 0.45, 0.55, 0.55, 0.20 ]
                 }
    
    # Frame-level to video-level aggregation of probabilities and ground truth
    y_prb_vid = frame2video(y_prb_frm, 'probability')
    y_gdt_vid = frame2video(y_gdt_frm, 'ground_truth') 
   
    # Calculate the equal error rate 
    # At frame-level
    eer_score_frm, eer_threshold_frm = eer(y_gdt_frm, y_prb_frm)
    # At video-level
    eer_score_vid, eer_threshold_vid = eer(y_gdt_vid, y_prb_vid)
    
    # Predictions using the eer_threshold
    # At frame-level
    y_prd_frm = prob2class(y_prb_frm, eer_threshold_frm)
    # At video-level
    y_prd_vid = prob2class(y_prb_vid, eer_threshold_vid)
    

    print('** Frame-level analysis **')
    print('Ground truth : ', y_gdt_frm)
    print('Probabilities of being fake (detector output) : ', y_prb_frm)
    print('')
    print('AUC :', f'{auc(y_gdt_frm, y_prb_frm):.3f}')
    print('EER :', f'{eer_score_frm:.3f}' )
    print('EER threshold :', f'{eer_threshold_frm:.3f}')
    print('')
    print('We apply the EER threshold to the vector probabilities to obtain ' +    'the vector of predicted classes')
    print('---')
    print('Predicted classes : ', y_prd_frm)
    print('Accuracy : ', f'{acc(y_gdt_frm, y_prd_frm):.3f}')
    print('F1-score : ', f'{f1(y_gdt_frm, y_prd_frm):.3f}')
    print('')
    roc(y_gdt_frm, y_prb_frm)

    
    print('** Image-level analysis **')
    print('Ground truth : ', y_gdt_vid)
    print('Probabilities of being fake (detector output) : ', y_prb_vid)
    print('')
    print('AUC :', f'{auc(y_gdt_vid, y_prb_vid):.3f}')
    print('EER :', f'{eer_score_vid:.3f}' )
    print('EER threshold :', f'{eer_threshold_vid:.3f}')
    print('')
    print('We apply the EER threshold to the vector probabilities to obtain ' +    'the vector of predicted classes')
    print('---')
    print('Predicted classes : ', y_prd_vid)
    print('Accuracy : ', f'{acc(y_gdt_vid, y_prd_vid):.3f}')
    print('F1-score : ', f'{f1(y_gdt_vid, y_prd_vid):.3f}')
    roc(y_gdt_vid, y_prb_vid)

    
