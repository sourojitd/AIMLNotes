# 📚 My Advanced Machine Learning Notes

---

## 앙 Ensemble Methods

### **1. Core Concepts**

**Detailed Explanation:**
Ensemble methods are techniques that combine the predictions from multiple separate models (called base estimators or weak learners) to achieve better performance and robustness than any single model could. The core motivation is the belief that a "committee of experts" working together is more likely to be accurate than any individual expert.

For an ensemble to be effective, two key conditions must be met:
1.  The base models should be as **different** from each other as possible.
2.  The errors made by each model should be **independent** (i.e., they should make different mistakes).

**Prediction Aggregation:**
* **For Classification:** The final prediction is the **Mode** (the most common vote) of the predictions from all the models.
* **For Regression:** The final prediction is the **average** of the predictions from all the models.

> #### 🧠 Kid Card: The Committee of Experts
>
> Imagine you have a very hard math problem. Instead of asking just one smart friend, you ask 5 different smart friends (your *models*).
>
> * Friend A, B, and C all get "10".
> * Friend D gets "9".
> * Friend E gets "11".
>
> You'd probably trust the most common answer, "10". This "committee" of friends is an **ensemble**. By combining their answers, you're less likely to be wrong than if you just trusted Friend D.

### **2. Bagging (Bootstrap Aggregating)**

**Detailed Explanation:**
Bagging is an ensemble method that builds independent models in **parallel**. Each model is trained on a random sample of the data, and their predictions are combined. Each "weak learner" has equal weight in the final prediction.

The training process involves **random sampling with replacement** (also called bootstrapping). This means that for a dataset of size 'n', a new sample is created by randomly picking 'n' data points, but after a point is picked, it's "put back" into the dataset, so it could be picked again. This ensures the samples are independent.

A neat side effect of this is that each sample (or "bag") contains about **63%** of the original data. The remaining **~37%** that were left out are called **"Out-of-Bag" (OOB)** samples. This OOB data can be used as a free validation set to test the model's performance. The main benefit of bagging is that it helps to **reduce the variance** of the model.

> #### 🧠 Kid Card: The Random Handfuls
>
> Imagine you have a big bag of 100 different LEGOs (your data). You want to build 5 different spaceships (your models).
>
> * **For Ship 1:** You reach in and pull out 100 LEGOs *with replacement* (meaning after you pull one out, you look at it, then put it back). Your handful will have some duplicates and be missing others.
> * **For Ship 2:** You do it again. You get another, slightly different handful of 100.
>
> You do this 5 times, so you have 5 different handfuls of LEGOs. You build one ship from each handful. Now you have 5 different ships!
>
> This process is called **Bagging**. The variety makes your final "fleet" of models much stronger and less likely to make a single, silly mistake.

### **3. Random Forest**

**Detailed Explanation:**
A Random Forest is a popular, more advanced version of bagging. It uses all the concepts of bagging (sampling with replacement, parallel training) but adds one extra layer of randomness to make the models (which are decision trees) even more different from each other.

* **Key Difference:** In a normal bagging decision tree, when deciding how to split a node, the tree considers *all* available features. In a Random Forest, it only considers a *random subset* of features ($m$) at each split. For example, if you have 20 features, the tree might only be allowed to look at 5 random ones to make its decision.

The trees are then grown to their maximum possible size. The final prediction is made by aggregating all the trees' predictions (mode for classification, average for regression). Key hyperparameters include `n_estimators` (the number of trees) and `max_features` (the number of features $m$ to consider at each split).

> #### 🧠 Kid Card: The "Limited-Clues" Forest
>
> This is like Bagging, but *even more* random.
>
> You're still building lots of decision trees, each with a "random handful" of data.
>
> But you also add a new rule: When each tree is growing, at every "fork in the road" (a node split), it's not allowed to look at *all* the clues (features). It has to randomly pick just 3 clues (e.g., "color," "shape," "size," but not "weight" or "texture") and make its decision *only* based on those.
>
> This forces each tree to be creative and find different paths to the answer. This extra variety makes the final "forest" of trees incredibly smart and hard to fool.

### **4. Boosting (Sequential Learning)**

**Detailed Explanation:**
Boosting is a different kind of ensemble. Instead of building models in parallel, it builds them **sequentially**. The core idea is that each new model attempts to **correct the errors** made by the previous models.

This is achieved through **weighted learning**. Data points that were misclassified by the previous model are given *more weight*. This forces the *next* model to pay extra attention to those specific "hard" data points. The final prediction is a weighted average or vote of all the learners. The main benefit of boosting is that it helps to **reduce the bias** of the model.

> #### 🧠 Kid Card: The Relay Race Team
>
> Boosting is like building a relay race team, one runner at a time.
>
> 1.  **Runner 1 (Model 1):** Runs the race. They do pretty well, but they get 5 questions wrong.
> 2.  **Runner 2 (Model 2):** Is a "specialist." Their *only* job is to focus on the 5 questions Runner 1 got wrong. They get most of them right, but still get 2 wrong.
> 3.  **Runner 3 (Model 3):** Is an even bigger specialist. Their only job is to fix the 2 mistakes Runner 2 left behind.
>
> You keep adding specialist runners to your team, each one learning from the mistakes of the person before. The final team is super accurate because everyone's mistakes were corrected by someone else.

### **5. AdaBoost (Adaptive Boosting)**

**Detailed Explanation:**
AdaBoost is one of the most popular boosting algorithms. It works exactly as described above:
1.  It starts by giving every sample an **equal weight**.
2.  It builds a "weak learner" (usually a very simple decision tree, often just one split).
3.  It measures the learner's importance based on its errors.
4.  It **increases the weights** for all the samples that were misclassified and decreases the weights for the ones that were correct.
5.  It builds the next weak learner, which now pays more attention to the newly weighted "hard" samples.
6.  This repeats for a specified number of estimators (`n_estimators`). The final prediction is a weighted vote of all the learners.

> #### 🧠 Kid Card: The "Hard-Question" Spotlight
>
> AdaBoost is a boosting game where you have a "difficulty" spotlight.
>
> * **Round 1:** All questions have a "1x" difficulty. Your first model tries to answer them. It gets 3 hard ones wrong.
> * **Round 2:** AdaBoost shines a "10x" spotlight on those 3 hard questions. The next model *must* pay attention to them because they're worth more points. It gets 2 of them right, but misses a new one.
> * **Round 3:** The spotlight now moves. The 2 hard questions go back to "1x," but the new one it missed gets the "10x" spotlight.
>
> It keeps "adapting" the difficulty to force new models to focus on the biggest mistakes.

### **6. Gradient Boosting (GBM)**

**Detailed Explanation:**
Gradient Boosting is another prominent boosting algorithm, but it has a clever twist on "learning from errors".

* **Key Difference:** Instead of changing the *weights* of the samples (like AdaBoost), Gradient Boosting fits the *next* weak learner directly to the **residuals** of the *previous* learner.
* A **residual** is simply the error (e.g., `actual_value - predicted_value`).
* **Process:**
    1.  A weak learner is built.
    2.  The errors (residuals) are calculated for every data point.
    3.  These residuals **become the new target variable** for the *next* model.
    4.  The next model is trained to *predict the error* of the previous model.
    5.  This process is repeated, with each new model trying to predict the leftover residuals, until the errors are minimized.

> #### 🧠 Kid Card: The "Guess-the-Mistake" Game
>
> Gradient Boosting is a different kind of team game.
>
> * **Player 1:** Tries to guess everyone's weight.
    * Guesses 150 lbs for Ali (who is 160). The **error is +10 lbs**.
    * Guesses 135 lbs for Bo (who is 130). The **error is -5 lbs**.
> * **Player 2:** Their job is *not* to guess the weights. Their job is to guess the *mistakes* from Player 1.
    * They try to predict "+10" for Ali and "-5" for Bo. Maybe they guess "+8" for Ali and "-6" for Bo.
> * **Player 3:** Their job is to guess the mistakes Player 2 *still* made.
    * (The new error for Ali is +2, for Bo it's -1).
>
> Your final prediction is: **Player 1's guess + Player 2's guess + Player 3's guess...** Each player is just cleaning up the "leftover error" from the one before.

### **7. XGBoost (Extreme Gradient Boosting)**

**Detailed Explanation:**
XGBoost is an upgraded, high-performance implementation of Gradient Boosting. Its main focus is on **computational speed** and **model performance**.

It achieves this speed through advanced features:
* **Parallelization:** Data is stored in "blocks" that allow for finding the best split in a tree in parallel.
* **Cache Optimization:** It makes the best use of your computer's hardware.
* **Out-of-Core Computing:** It can handle massive datasets that don't even fit in your computer's memory.
* **Missing Values:** It is designed to handle missing values internally by learning a default direction for them during training.

It also has special hyperparameters to prevent overfitting, such as `gamma` (the minimum loss reduction required to make a split) and `colsample_bytree` (using only a fraction of columns to build each tree).

> #### 🧠 Kid Card: The Gradient Boosting Racecar
>
> If Gradient Boosting is a smart game, **XGBoost** is that same game played in a *Formula 1 racecar*.
>
> It does the same "guess-the-mistake" job, but it's built to be **extremely fast** and **efficient**.
>
> * It can use all your computer's power at once (parallelization).
> * It can work with datasets so huge they don't fit in memory.
> * It doesn't even crash if you have missing data; it *learns* what to do with it.
>
> It's the most popular version because it's so fast, smart, and powerful.

### **8. Stacking**

**Detailed Explanation:**
Stacking is an advanced ensemble that combines **heterogeneous models** (models of different types, like KNN, Logistic Regression, and SVM).

Instead of a simple vote, it uses a "meta-model" to learn how to best combine the predictions.
* **Process:**
    1.  The base models (e.g., KNN, SVM) are trained.
    2.  The predictions from these base models are saved.
    3.  These predictions are then used as the **input features** for a final "meta-model".
* This meta-model (often a simple logistic regression) literally *learns* how to combine the base model predictions to get the best result. It might learn that the SVM is very trustworthy, but the KNN is only right about 50% of the time, and it will adjust its final answer based on that.

> #### 🧠 Kid Card: The All-Star Team and the Manager
>
> Stacking is like building an All-Star team with a manager.
>
> * **Level 1 (The Players):** You train a bunch of *different* kinds of models.
    * You have a "Random Forest" player (an expert at trees).
    * You have an "SVM" player (an expert at finding lines).
    * You have a "KNN" player (an expert at finding neighbors).
> * **Level 2 (The Manager):** You hire a final, new model called a "meta-model".
>
> To make a final decision, the players don't vote. They just give their *opinion* (their prediction) to the Manager.
>
> The Manager's *only job* is to learn which players to trust. It learns, "Okay, for this kind of problem, I trust the Forest and SVM, but I'll ignore the KNN." The **Manager** makes the final call.

---

## 🧐 Model Evaluation & Validation

### **1. Cross-Validation**

**Detailed Explanation:**
Cross-validation is a technique to estimate how well your model will perform on new, unseen data, especially when you don't have a massive dataset. The goal is to get a "generalized" performance score.

The most common method is **k-fold cross-validation**:
1.  The data is divided into $k$ equal segments, or "folds" (e.g., $k=10$).
2.  In the first run, the model is trained on folds 1-9, and tested on fold 10.
3.  In the second run, it's trained on folds 1-8 and 10, and tested on fold 9.
4.  This process is repeated $k$ times, so that each fold is used as the test set exactly once.
5.  You will get $k$ different performance scores (one from each run).
6.  The final model performance is the **average** of all $k$ scores. This gives you a much more reliable estimate of your model's true performance.

> #### 🧠 Kid Card: The 10-Page Pop Quiz
>
> Imagine you have a 10-page study guide for a big test, and you want to know how well you'll do.
>
> A *bad* way to study is to just read pages 1-9 and take a practice test on page 10. What if page 10 was super easy or super hard? You'd get a misleading score.
>
> **Cross-Validation is the *smart* way:**
> 1.  **Test 1:** You study pages 1-9, and take a test on **page 10**.
> 2.  **Test 2:** You study pages 1-8 & 10, and take a test on **page 9**.
> 3.  **Test 3:** You study pages 1-7 & 9-10, and take a test on **page 8**.
>
> ...you do this 10 times, so *every single page* gets to be the "test" exactly once.
>
> Your final grade is the **average of all 10 tests**. This is a much, much more trustworthy score of how well you know the material.

### **2. Underfitting & Overfitting**

**Detailed Explanation:**
* **Underfitting:** This occurs when a model is too simplistic to capture the real patterns in the data. You know you are underfitting when the model performs poorly *even on the training set*. This can be caused by low model complexity or using irrelevant features. To fix it, you can **increase model complexity** (e.g., use a more powerful model).

* **Overfitting:** This occurs when a model learns the training data *too well*, memorizing the noise and outliers. You know you are overfitting when the model gets a great score on the training data but a **terrible score on the test data** (e.g., 98% train accuracy, 55% test accuracy). This is caused by high model complexity or a small/noisy dataset. To fix it, you can use **regularization**, **get more data**, or **decrease model complexity**.

> #### 🧠 Kid Card: The Test Study Analogy
>
> * **Underfitting (Too Simple):** You're studying for a history test. You *only* read the chapter titles. You don't know the dates, the people, or the events. You fail the practice questions *and* you fail the real test. Your model (your brain) was "too simple".
>
> * **Overfitting (Too Specific):** You study by *memorizing the exact 50 questions* on the practice sheet, including the typos. You get 100% on the practice test! But when the *real test* asks the same questions in a slightly different way, you have no idea what to do and you fail. Your model "memorized the noise" instead of learning the *actual concepts*.

### **3. Data Leakage**

**Detailed Explanation:**
Data leakage is a critical error where information from the *test set* is unintentionally used during the *model training process*. This leads to an artificially high, unreliable performance on the test set, because the model has essentially "seen the answers".

Common causes include:
* **Standardizing (e.g., z-score) the *entire* dataset** before splitting it into train and test sets.
* **Imputing missing values for the *entire* dataset** before splitting.
* Tuning hyperparameters to get the best score on the *test set* instead of a validation set.

**Prevention:** The best way to prevent leakage is to **split your data first** and "keep a portion of the sample data away before doing any processing". All transformations (like standardizing) should be fit *only* on the training data, and then *applied* to the test data.

> #### 🧠 Kid Card: Cheating on the Spelling Bee
>
> Data leakage is just a fancy word for **cheating**.
>
> Imagine you're in a spelling bee. You get a list of 100 'practice words' (your **train set**) and you'll be tested on 10 'secret words' (your **test set**).
>
> **Data Leakage is...** your friend finds the 'secret' 10-word list and mixes it into your 100-word practice list.
>
> You practice all 110 words. When the real spelling bee happens, you get 100% on the secret words!
>
> You *look* like a genius (your model has "high accuracy"), but you didn't *actually* learn how to spell better. You just saw the answers ahead of time.

---

## 🛠️ Data Preprocessing & Tuning

### **1. Handling Imbalanced Data**

**Detailed Explanation:**
Imbalanced datasets are common in fields like banking or health, where one class (the "majority class") significantly outnumbers the other (the "minority class," e.g., 'fraud' or 'rare disease'). A model trained on this will become biased and just predict the majority class every time.

Strategies to balance the classes include:
* **Oversampling:** Increasing the frequency of the minority class. The simplest way is **Random Oversampling**, which just duplicates random records from the minority class (but this can lead to overfitting).
* **Undersampling:** Decreasing the frequency of the majority class. **Random Undersampling** removes random records from the majority class (but this can cause information loss).
* **SMOTE (Synthetic Minority Oversampling Technique):** A smarter way to oversample. It creates *new, synthetic* data points for the minority class. It does this by picking a minority point, finding its *k-nearest neighbors*, and then adding a new point somewhere on the line *between* the original point and its neighbors.
* **Tomek links:** These are pairs of data points that are very close but belong to *opposite* classes. This is an undersampling technique where the data point from the **majority class** in the pair is removed, which helps to create a cleaner border between the classes.

> #### 🧠 Kid Card: The "Bad Guy" Detector
>
> Imagine you're training a robot to spot a "bad guy" in a crowd. You give it 100 photos to learn from:
> * 99 photos of "Normal People" (Majority Class)
> * 1 photo of a "Bad Guy" (Minority Class)
>
> The robot learns a lazy rule: "Just say 'Normal Person' every time!" It will be 99% accurate, but it's useless. To fix this:
>
> * **Oversampling:** You put the "Bad Guy" photo in a photocopier 98 times. Now you have 99 "Normal" and 99 "Bad Guy" photos. The robot is forced to learn.
> * **SMOTE (Smarter Oversampling):** Instead of just copying, you use a drawing program to create *new* "Bad Guy" photos that look *similar* to the original but are slightly different (e.g., different hat, new background). This gives the robot better examples.
> * **Undersampling:** You just throw away 98 of the "Normal People" photos. Now you have 1 "Normal" and 1 "Bad Guy". This is balanced, but your robot has very little to learn from.

### **2. Hyperparameter Tuning**

**Detailed Explanation:**
Hyperparameters are parameters that are **set before the learning process begins** and govern the entire training process. Examples include the `learning_rate` in boosting or the number of trees (`n_estimators`) in a Random Forest.

The process of finding the *optimal* hyperparameters for your model is called **hyperparameter tuning**. This is crucial because good hyperparameters can significantly improve model performance and help reduce both overfitting and underfitting.

The typical process involves:
1.  Choosing your model (e.g., `sklearn.svm.SVC()`).
2.  Defining the "parameter space" to search (e.g., `C: [0.1, 1, 10]`, `kernel: ['linear', 'rbf']`).
3.  Choosing a search method (like GridSearchCV).
4.  Choosing a cross-validation scheme.
5.  Choosing a score function to evaluate the results (like 'accuracy').

> #### 🧠 Kid Card: The Oven Knobs
>
> If your model is an "oven" and your data is the "cookie dough," then...
>
> * The "parameters" are the things the oven *learns* by itself while baking (like how the heat spreads).
> * The **"hyperparameters"** are the *knobs on the outside* that *you* must set *before* you start baking:
    * `Temperature`
    * `Time`
    * `Convection Fan On/Off`
>
> **Hyperparameter Tuning** is the process of trying a bunch of different knob combinations (e.g., "350° for 10 min" vs. "400° for 8 min") to find the *single best recipe* that makes the most perfect cookie.

### **3. Hyperparameter Tuning Methods**

**Detailed Explanation:**
Finding the best hyperparameters can be very time-consuming. The two most common methods are:

* **GridSearchCV (Grid Search):** This technique **exhaustively considers all parameter combinations** you provide in a "grid". For each combination, it runs a cross-validation and records the score. Finally, it tells you which combination had the best score.
    * **Pro:** It will find the best combination *in your grid*.
    * **Con:** It is very slow and computationally expensive, especially for large search spaces.

* **RandomizedSearchCV (Random Search):** This technique is often faster and more efficient. Instead of trying *all* combinations, it **samples a fixed number of random candidates** (given by `n_iter`) from the parameter space.
    * **Pro:** It is much faster and works well on large search spaces. Research shows that for many models, only a few hyperparameters actually matter, and a random search has a higher chance of finding a good combination of those important parameters.

> #### 🧠 Kid Card: Finding the Best Ice Cream
>
> You want to find the best-tasting combination of 10 ice cream `flavors` and 10 `toppings`. That's 100 possible combos.
>
> * **Grid Search (The "Try-Everything" Method):** You must patiently eat all 100 combinations (Vanilla+Sprinkles, Vanilla+Fudge... Chocolate+Sprinkles... all the way to Mint+Caramel). You are *guaranteed* to find the absolute best one, but it will take all day and make you sick.
>
> * **Random Search (The "Lucky-Dip" Method):** You only have time to try 10 combinations. So you randomly pick 10 pairs from the 100 possibilities (e.g., "Mint+Fudge," "Strawberry+Peanuts," "Coffee+Sprinkles"...). You might not find the *absolute perfect* combo, but you'll probably find a *really good* one, and it was 10 times faster.
