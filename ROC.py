import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

fpr_Alex_wo_feature, tpr_Alex_wo_feature = np.loadtxt('Alex_wo_feature.txt',delimiter=' ')
roc_auc_Alex_wo_feature = metrics.auc(fpr_Alex_wo_feature, tpr_Alex_wo_feature)

fpr_Alex_wo, tpr_Alex_wo = np.loadtxt('Alex_wo.txt',delimiter=' ')
roc_auc_Alex_wo = metrics.auc(fpr_Alex_wo, tpr_Alex_wo)

fpr_Alex_w_feature, tpr_Alex_w_feature = np.loadtxt('Alex_w_feature.txt',delimiter=' ')
roc_auc_Alex_w_feature = metrics.auc(fpr_Alex_w_feature, tpr_Alex_w_feature)

fpr_Alex_w_feature_improve, tpr_Alex_w_feature_improve = np.loadtxt('Alex_w_feature_improve.txt',delimiter=' ')
roc_auc_Alex_w_feature_improve = metrics.auc(fpr_Alex_w_feature_improve, tpr_Alex_w_feature_improve)

fpr_VGG16_wo_feature, tpr_VGG16_wo_feature = np.loadtxt('VGG16_wo_feature.txt',delimiter=' ')
roc_auc_VGG16_wo_feature = metrics.auc(fpr_VGG16_wo_feature, tpr_VGG16_wo_feature)

fpr_VGG16_wo, tpr_VGG16_wo = np.loadtxt('VGG16_wo.txt',delimiter=' ')
roc_auc_VGG16_wo = metrics.auc(fpr_VGG16_wo, tpr_VGG16_wo)

fpr_VGG16_w_feature_improve, tpr_VGG16_w_feature_improve = np.loadtxt('VGG16_w_feature_improve.txt',delimiter=' ')
roc_auc_VGG16_w_feature_improve = metrics.auc(fpr_VGG16_w_feature_improve, tpr_VGG16_w_feature_improve)

fig=plt.figure()
lw = 2
plt.plot(
    fpr_Alex_wo_feature,
    tpr_Alex_wo_feature,
    color="darkorange",
    lw=lw,
    label="AlexNet w/o feature_layer (AUC = %0.2f)" % roc_auc_Alex_wo_feature,
)
# plt.plot(
#     fpr_Alex_wo,
#     tpr_Alex_wo,
#     color="maroon",
#     lw=lw,
#     label="AlexNet w/o  (AUC = %0.2f)" % roc_auc_Alex_wo,
# )
# plt.plot(
#     fpr_Alex_w_feature,
#     tpr_Alex_w_feature,
#     color="seagreen",
#     lw=lw,
#     label="AlexNet w/ (AUC = %0.2f)" % roc_auc_Alex_w_feature,
# )

plt.plot(
    fpr_Alex_w_feature_improve,
    tpr_Alex_w_feature_improve,
    color="deepskyblue",
    lw=lw,
    label="AlexNet w/ feature_layer (AUC = %0.2f)" % roc_auc_Alex_w_feature_improve,
)

plt.plot(
    fpr_VGG16_wo_feature,
    tpr_VGG16_wo_feature,
    color="royalblue",
    lw=lw,
    label="VGG-16 w/o feature_layer (AUC = %0.2f)" % roc_auc_VGG16_wo_feature,
)

# plt.plot(
#     fpr_VGG16_wo,
#     tpr_VGG16_wo,
#     color="orchid",
#     lw=lw,
#     label="VGG-16 w/o (AUC = %0.2f)" % roc_auc_VGG16_wo,
# )

plt.plot(
    fpr_VGG16_w_feature_improve,
    tpr_VGG16_w_feature_improve,
    color="violet",
    lw=lw,
    label="VGG-16 w/ feature_layer (AUC = %0.2f)" % roc_auc_VGG16_wo_feature,
)

plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve of LFW pairs Dataset")
plt.legend(loc="lower right")
plt.show()

fig.savefig('ROC_fine_tunning.png')