from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                            precision_score, recall_score, roc_auc_score, roc_curve,
                            classification_report
                            )
import matplotlib.pyplot as plt
import numpy as np

#######################################################################
#---------------------------------------------------------------------#
#---------------   PLOT AND METRIC FUNCTIONS   -----------------------#
#---------------------------------------------------------------------#
#######################################################################


def plot_acc_loss(history,title,fig_path,to_save=False):
    
    fig = plt.figure(figsize=(15,5))
    
    plt.suptitle(title, fontsize=16)
    
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])        
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    #plt.show() 
    
    if to_save:
        fig.savefig(fig_path + 'accuracy_loss_vs_epoch.png')

#--------------------------------------------------------------------------------
        
def plot_metrics_vs_epoch(history,fig_path,history_dot_history=False,to_save=False):
    
    fig = plt.figure(figsize=(12,8))
    if history_dot_history:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc']) 
    else:
        plt.plot(history['acc'])
        plt.plot(history['val_acc']) 
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    #plt.show() 

    if to_save:
        fig.savefig(fig_path + 'accuracy_vs_epochs.png')

    fig = plt.figure(figsize=(12,8))
    if history_dot_history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss']) 
    else:
        plt.plot(history['loss'])
        plt.plot(history['val_loss']) 
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    #plt.show()

    if to_save:
        fig.savefig(fig_path + 'loss_vs_epochs.png')    
    
    fig = plt.figure(figsize=(12,8))
    if history_dot_history:
        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc']) 
    else:
        plt.plot(history['auc'])
        plt.plot(history['val_auc']) 
    plt.title('AUC vs. epochs')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    #plt.show()
    
    if to_save:
        fig.savefig(fig_path + 'auc_vs_epochs.png')
        
    fig = plt.figure(figsize=(12,8))
    if history_dot_history:
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall']) 
    else:
        plt.plot(history['recall'])
        plt.plot(history['val_recall']) 
    plt.title('Recall vs. epochs')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    #plt.show() 

    if to_save:
        fig.savefig(fig_path + 'recall_vs_epochs.png')
    
    fig = plt.figure(figsize=(12,8))
    if history_dot_history:
        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision']) 
    else:
        plt.plot(history['precision'])
        plt.plot(history['val_precision']) 
    plt.title('Precision vs. epochs')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    #plt.show() 

    if to_save:
        fig.savefig(fig_path + 'precision_vs_epochs.png')

#--------------------------------------------------------------------------------
def plot_sample_images_onehot_labels(folders,batch_images,batch_labels,title,fig_path,to_save=False):
    
    # classes are encoded as one-hot labels: image_generator.flow_from_directory has class_mode='categorical'
    
    fig = plt.figure(figsize=(14,12))
    plt.suptitle(title, fontsize=20)
    
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(batch_images[i], cmap='gray')
        plt.title(folders[np.where(batch_labels[i] == 1.)[0][0]])
        plt.axis('off')
        
    if to_save:
        fig.savefig(fig_path + 'sample_images.png')
        
#--------------------------------------------------------------------------------
def plot_sample_images(folders,batch_images,batch_labels,title,fig_path,to_save=False):
    
    # classes are encoded as sparse labels (integers): image_generator.flow_from_directory has class_mode='sparse'
    
    fig = plt.figure(figsize=(14,12))
    plt.suptitle(title, fontsize=20)
    
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(batch_images[i], cmap='gray')
        plt.title(folders[int(batch_labels[i])])
        plt.axis('off')
        
    if to_save:
        fig.savefig(fig_path + 'sample_images.png')        
        
#--------------------------------------------------------------------------------        
def get_prediction_metrics(y_test,y_pred,fig_path,folders,to_save=False):
    
    msg = ""
    
    # Classification report
    cla_rep = classification_report(y_test,y_pred)
    msg += "Classification Report\n"
    msg += cla_rep
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    msg += "\nConfusion Matrix\n"
    msg +=  str(cm)
    
    #print("Confusion Matrix")
    #print(cm)
    
    # TP, FP, FN, TP rate
    tn, fp, fn, tp = cm.ravel()
    msg += '\nTN: %d, FP: %d, FN: %d, TP: %d\n'%(tn, fp, fn, tp)
    #print('TN: %d, FP: %d, FN: %d, TP: %d'%(tn, fp, fn, tp))
    # In binary classification, cm[0,0]=TN, cm[0,1]= FP, cm[1,0]= FN, cm[1,1]=TP

    # Metrics
    ac = accuracy_score(y_test,y_pred) 
    pre = precision_score(y_test,y_pred)  
    rec = recall_score(y_test,y_pred) 
    roc_auc = roc_auc_score(y_test,y_pred) 
    
    msg += "Accuracy %.2f, Precision %.2f, Recall %.2f, AUC %.2f\n"%(ac,pre,rec,roc_auc)
    #print("Accuracy %.2f, Precision %.2f, Recall %.2f, AUC %.2f"%(ac,pre,rec,roc_auc))
    
    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=folders)
    disp.plot(ax=ax,cmap='Blues')
    plt.tight_layout()
    #plt.show()
    
    if to_save:
        fig.savefig(fig_path + 'confusion_matrix.png')
        print('==> Confusion matrix saved at: ' + str(fig_path) + 'confusion_matrix.png')
    
    # Plot ROC Curve   
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    #print('fpr: ', fpr)
    #print('tpr: ', tpr)
    fig = plt.figure(figsize=(12,8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(roc_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.show()
    if to_save:
        fig.savefig(fig_path + 'roc_curve.png')

    print('==> '+ msg)
    
    return(msg)
#--------------------------------------------------------------------------------