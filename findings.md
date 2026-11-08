# Findings — Credit Card Customer Attrition

Working draft; final tables go into README.md.

## Comparison of evaluation metrics

| ML Model Name          | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------------------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression    | 0.9161   | 0.9316 | 0.8189 | 0.6123 | 0.7007 | 0.6624 |
| Decision Tree          | 0.9368   | 0.9476 | 0.8456 | 0.7415 | 0.7902 | 0.7554 |
| KNN                    | 0.9156   | 0.9131 | 0.8632 | 0.5631 | 0.6816 | 0.6547 |
| Naive Bayes            | 0.8697   | 0.8532 | 0.6000 | 0.5631 | 0.5810 | 0.5043 |
| Support Vector Machine | 0.9442   | 0.9657 | 0.8581 | 0.7815 | 0.8180 | 0.7864 |

## Observations

| ML Model Name                     | Observation about model performance |
|-----------------------------------|-------------------------------------|
| Logistic Regression               | Train and test scores are almost the same, so there is no overfitting. AUC is good (0.932), but recall is 0.612, so it misses about 39% of churned customers. As a linear model, it cannot capture combinations of features (for example, low transaction count together with low amount). |
| Decision Tree                     | Second best model (F1 0.790, MCC 0.755). The default tree overfit, so depth was limited to 6 using CV. This improved test AUC from 0.88 to 0.95. Recall (0.742) is much better than Logistic Regression. |
| KNN                               | Accuracy is high (0.916), but recall for churned customers is only 0.563, so it misses about 44% of them. Precision is good (0.863): when it predicts churn, it is usually correct. It has the highest precision but the lowest recall, F1 and MCC among the models so far. |
| Naive Bayes                       | Lowest scores among the models so far (accuracy 0.870, F1 0.581, MCC 0.504). Recall (0.563) is the same as KNN, but precision is only 0.600, so 4 out of 10 churn predictions are wrong. It assumes features are independent, but many features here are related (e.g. transaction amount and count), which is likely why it does worse. |
| Support Vector Machine            | Best scores of all models (accuracy 0.944, AUC 0.966, F1 0.818, MCC 0.786). Highest recall (0.782): it finds about 78% of churned customers. Precision (0.858) is close to the best (KNN, 0.863). C=10 and gamma="scale" were chosen with 5-fold CV. |
| Overall, Winner for your dataset? | SVM is the winner. It has the highest accuracy, AUC, recall, F1 and MCC. Decision Tree is second (F1 0.790, MCC 0.755). Naive Bayes is the weakest (F1 0.581). Since only ~16% of customers churn, we compare mainly on F1 and MCC, not accuracy. |
