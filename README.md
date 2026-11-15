# Enhancing Coral Bleaching Prediction Using Two-Part Machine Learning Models

**Authors:** Sumedh Kota, Pranav Kulkarni  
**Status:** Preprint - Submitted to International Conference on Advanced Research in Computing (ICARC) (Under Review)

📄 **[Download Full Paper (PDF)](./manuscript.pdf)**

---

## Abstract
Coral bleaching, driven largely by elevated sea surface temperatures, threatens reef biodiversity and ecosystems, and reliable prediction can help target monitoring and conservation. Therefore, we evaluate direct regression and two-part occurrence-severity models for predicting coral bleaching at previously unseen locations. Using 23,263 survey samples from the BCO-DMO global bleaching dataset (1980-2020), we grouped samples into 10,846 location groups based on site and reef identifiers and coordinates, with five outer grouped folds and three inner grouped folds separating evaluation from hyperparameter and threshold selection. The hard two-part model achieved the lowest mean absolute error (MAE) of 7.67 percentage points (95\% component-bootstrap interval: 7.37-7.97), compared with 12.86 for the training-median baseline and 8.22 for direct absolute-error boosting. Extra Trees had a comparable MAE of 7.73 percentage points and performed better on RMSE, group-macro MAE, and severe bleaching above 50\% bleaching. The occurrence classifier achieved a ROC AUC of 0.916, and the hard two-part model sharply reduced error on zero-bleaching samples but underpredicted severe events. Future work should test these models in distant regions and later years and extend the framework toward forecasting with temporally structured data. Two-part occurrence-severity modeling matched the best direct regression model overall and gave the lowest error on low-bleaching and zero-bleaching samples, supporting its effectiveness for predicting bleaching at unseen locations.

## Repository Contents
* `manuscript.pdf` - Full text of the paper (preprint version).
* `/code` - Script used to run the experiment, train the model, and generate figures.
