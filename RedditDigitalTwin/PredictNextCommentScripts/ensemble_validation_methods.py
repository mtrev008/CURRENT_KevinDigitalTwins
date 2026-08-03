import pandas as pd
import sys
sys.path.append('../')
from utilities import get_cosine_similarity, get_jaccard_similarity, F1_Score_Recall_Precision, get_majority_vote, get_readability_score
from open_source_llm import make_inference_call
import GoogleGemini as gemini
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from itertools import combinations
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

def majority_vote_classifier(models_predictions, true_labels):
    """Take the majority across all models and assigns a prediction. """
    model_names = list(models_predictions.keys())

    # shape: (num_examples, num_models)
    X = np.array(list(models_predictions.values())).T
    y = np.array(true_labels)
    y_pred = []

    for i in X:
        # count how many models predict 1 for each example
        vote_sum = np.sum(i)
        if vote_sum >= (len(model_names) / 2):
            pred = 1
        else: 
            pred = 0
        y_pred.append(pred)

    return y_pred


def logistic_regression_classifier_cv(true_labels, X=None, models_predictions=None, debug=False, n_splits=5):
    """Creates an ensemble of LLM predictions using logistic regression with cross-validation."""

    if X is not None:
        X = np.array(X) # For post embeddings only
    elif models_predictions is not None:
        X_df = pd.DataFrame(models_predictions)
        X = X_df.to_numpy()
    else:
        print("ERROR: Must provide either X or models_predictions.")

    y = np.array(true_labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    all_y_true = []
    all_y_pred = []

    scores = {
        "f1": [],
        "precision": [],
        "recall": [],
        "balanced_accuracy": [],
        "kappa": []
    }

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(penalty='l2', C=0.5, solver="lbfgs", random_state=2, max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        if debug:
            print("Results for current iteration:")
            print_results("Logistic regression", y_test, y_pred)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        scores["f1"].append(f1_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))
        scores["kappa"].append(cohen_kappa_score(y_test, y_pred))

    return all_y_true, all_y_pred, scores

def logistic_regression_hyperparameter_tuning(models_predictions, true_labels, debug=False, n_splits=5):
    """Conduct manual grid search over logistic regression parameters using cross-validation. Find the best combination of parameters. """

    X_df = pd.DataFrame(models_predictions)
    X = X_df.to_numpy()
    y = np.array(true_labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    # Set all parameters to test
    C_values = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    penalty_values = ["l2"]
    solver_values = ["liblinear", "lbfgs", "saga"]
    class_weight_values = [None, "balanced"]

    results = []

    for C in C_values:
        for penalty in penalty_values:
            for solver in solver_values:
                for class_weight in class_weight_values:

                    all_y_true = []
                    all_y_pred = []

                    for train_idx, test_idx in skf.split(X, y):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        clf = LogisticRegression(C=C, penalty=penalty, solver=solver, class_weight=class_weight, random_state=2, max_iter=1000)

                        clf.fit(X_train_scaled, y_train)
                        y_pred = clf.predict(X_test_scaled)

                        all_y_true.extend(y_test.tolist())
                        all_y_pred.extend(y_pred.tolist())

                    f1 = f1_score(all_y_true, all_y_pred)
                    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
                    recall = recall_score(all_y_true, all_y_pred, zero_division=0)
                    kappa = cohen_kappa_score(all_y_true, all_y_pred)

                    results.append({
                        "C": C,
                        "penalty": penalty,
                        "solver": solver,
                        "class_weight": class_weight,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "kappa": kappa
                    })

                    if debug:
                        print("Params:")
                        print("C:", C)
                        print("penalty:", penalty)
                        print("solver:", solver)
                        print("class_weight:", class_weight)
                        print("F1:", f1)
                        print("Precision:", precision)
                        print("Recall:", recall)
                        print("Kappa:", kappa)
                        print()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False).reset_index(drop=True)

    best_row = results_df.iloc[0]

    best_params = {
        "C": best_row["C"],
        "penalty": best_row["penalty"],
        "solver": best_row["solver"],
        "class_weight": best_row["class_weight"],
    }

    best_score = best_row["f1"]

    print("\nBest Logistic Regression Params:")
    print(best_params)
    print("Best F1:", best_score)
    print("Precision:", best_row["precision"])
    print("Recall:", best_row["recall"])
    print("Cohen's kappa:", best_row["kappa"])

    return results_df, best_params


def svm_linear_classifier_cv(X, true_labels, n_splits=5):
    X = np.array(X)
    y = np.array(true_labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    all_y_true = []
    all_y_pred = []

    scores = {
        "f1": [],
        "precision": [],
        "recall": [],
        "balanced_accuracy": [],
        "kappa": []
    }

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LinearSVC(class_weight="balanced", random_state=2, max_iter=10000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        scores["f1"].append(f1_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))
        scores["kappa"].append(cohen_kappa_score(y_test, y_pred))

    return all_y_true, all_y_pred, scores

def random_forest_classifier(X, true_labels, n_splits=5):
    """Makes predictions based on input post embeddings. """

    X = np.array(X)
    y = np.array(true_labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    all_y_true = []
    all_y_pred = []
    
    scores = {
        "f1": [],
        "precision": [],
        "recall": [],
        "balanced_accuracy": [],
        "kappa": []
    }

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=2
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        scores["f1"].append(f1_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))
        scores["kappa"].append(cohen_kappa_score(y_test, y_pred))

    return all_y_true, all_y_pred, scores

def gradient_boosting_classifier(X, true_labels, n_splits=5):
    """Makes predictions based on input post embeddings using gradient boosting."""

    X = np.array(X)
    y = np.array(true_labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    all_y_true = []
    all_y_pred = []

    scores = {
        "f1": [],
        "precision": [],
        "recall": [],
        "balanced_accuracy": [],
        "kappa": []
    }

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=2
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        scores["f1"].append(f1_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))
        scores["kappa"].append(cohen_kappa_score(y_test, y_pred))

    return all_y_true, all_y_pred, scores

def xgboost_classifier(X, true_labels, n_splits=5):
    """Makes predictions based on input post embeddings using XGBoost."""

    X = np.array(X)
    y = np.array(true_labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    all_y_true = []
    all_y_pred = []

    scores = {
        "f1": [],
        "precision": [],
        "recall": [],
        "balanced_accuracy": [],
        "kappa": []
    }

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=2
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        scores["f1"].append(f1_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))
        scores["kappa"].append(cohen_kappa_score(y_test, y_pred))

    return all_y_true, all_y_pred, scores

def compute_agreement_matrix(models_predictions):
    model_names = list(models_predictions.keys())
    n = len(model_names)

    agreement_matrix = pd.DataFrame(np.zeros((n, n)), index=model_names, columns=model_names)

    for i in range(n):
        for j in range(n):
            preds_i = np.array(models_predictions[model_names[i]])
            preds_j = np.array(models_predictions[model_names[j]])
            agreement = np.mean(preds_i == preds_j)
            agreement_matrix.iloc[i, j] = agreement

    return agreement_matrix


def run_logistic_grid_search(models_predictions, true_labels, top_k=10):
    """Try all combinations of learners with logistic regression cross validation."""

    learner_names = list(models_predictions.keys())
    results = []

    for r in range(1, len(learner_names) + 1):
        for combo in combinations(learner_names, r):
            selected_predictions = {}
            for learner in combo:
                selected_predictions[learner] = models_predictions[learner]

            y_true, y_pred = logistic_regression_classifier_cv(true_labels, models_predictions=selected_predictions, n_splits=5)

            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            kappa = cohen_kappa_score(y_true, y_pred)

            results.append({
                "learners": combo,
                "num_learners": len(combo),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "kappa": kappa
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False).reset_index(drop=True)

    print(f"Top {top_k} logistic regression ensembles:\n")
    for i in range(min(top_k, len(results_df))):
        row = results_df.iloc[i]
        print(f"Rank {i+1}")
        print("Learners:", list(row["learners"]))
        print("F1 Score:", row["f1"])
        print("Precision:", row["precision"])
        print("Recall:", row["recall"])
        print("Cohen kappa score:", row["kappa"])
        print()

    return results_df

def print_results(name, y_true, y_pred):
    print("\n" + name)
    print("F1:", f1_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
    print("Cohen's kappa:", cohen_kappa_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

def main():
    # Setup variables
    debug=False

    # Init Google Gemini
    gemini.InitGoogleGemini(free_tier=False)

    # Set up ground truth labels
    annotations_df = pd.read_csv("AdditionalAnnotators_DigitalTwins_PostPairAnnotations - TrueBias.csv") # Enter correct annotations filepath
    # Use the following if there are 3 annotators
    # groundtruth_true_df = get_majority_vote(annotations_df)
    # groundtruth_true_df['majority_vote'] = groundtruth_true_df['majority_vote'].astype(str)
    # groundtruth_true_df['majority_vote'] = groundtruth_true_df['majority_vote'].str.lower() # Lowercase each entry in column
    # groundtruth_true_df['majority_vote'] = groundtruth_true_df['majority_vote'].map({'true': 1, 'false': 0}).fillna(0)

    # groundtruth_df = groundtruth_true_df.copy()
    # print(groundtruth_df.columns)
    # print("Len of ground truth:", len(groundtruth_df))

    # true_labels = groundtruth_df['majority_vote'].to_list()

    # Temporary only for my annotations, delete later
    groundtruth_df = annotations_df.copy()
    groundtruth_df['annotator_1'] = groundtruth_df['annotator_1'].astype(str).str.lower()
    groundtruth_df['annotator_1'] = groundtruth_df['annotator_1'].map({'true': 1, 'false': 0}).fillna(0)
    true_labels = groundtruth_df['annotator_1'].to_list()

    ################## Get agentic result DFs #################
    gemini_judge_df = pd.read_csv("AgentResults/Gemini_Llama_independent_judge_on_disagreement.csv") 
    gemini_and_llama_debate_df = pd.read_csv("AgentResults/Gemini_Llama_agentic_debate.csv")
    
    # judge_features_df = gemini_judge_df.drop(columns=["ground_truth"], errors="ignore")
    # judge_features_df = judge_features_df.drop(columns=["post_index"], errors="ignore")
    # judge_models_predictions = judge_features_df.to_dict(orient="list")

    ####################################################################

    # Get labels from LLMs:
    predicted_labels = []
    # cosine_similarities = []
    jaccard_sim = []
    post_distances = []
    fre_differences = []

    for i, post in enumerate(groundtruth_df['User Post']):
        LLM_output = groundtruth_df["LLM Output"][i] # LLM's post
        # Get cosine similarity
        sentences = [LLM_output, post]
        # similarity = get_cosine_similarity(sentences)
        # print(f"Similarity: {similarity}")
        # cosine_similarities.append(similarity)
        distance = len(LLM_output) - len(post)
        fre_diff = get_readability_score(LLM_output) - get_readability_score(post)
        fre_differences.append(fre_diff)
        jac_sim = get_jaccard_similarity(sentences)
        jaccard_sim.append(jac_sim)
        post_distances.append(distance)

    # Get cosine similarity of LLM and user outputs and write to csv file 
    # cosine_sim_for_groundtruth_posts = pd.DataFrame()
    # cosine_sim_for_groundtruth_posts['cosine_sim'] = cosine_similarities
    # cosine_sim_for_groundtruth_posts.to_csv('CosineSimForGroundtruthPosts.csv', index=False)
        
    # "Qwen/Qwen3-32B"
    models = ["gemini-3-flash-preview", "openai/gpt-oss-120b", "Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-R1", "meta-llama/Llama-3.3-70B-Instruct"]

    models_predictions = {}

    for model_name in models:
        print('Testing:', model_name)
        predicted_labels = [] # Reset predicted labels
        
        for i, post in enumerate(groundtruth_df['User Post']): 
            if debug:
                print("Analyzing post:", i)
            
            LLM_output = groundtruth_df["LLM Output"][i] # LLM's post

            prompt = ""
            prompt += "In essence, are the LLM-generated and the user-authored texts broadly making the same key points? "
            prompt += "Respond with either True or False. \n"
            prompt += "Focus on the BROAD key points, not narrow differences in wording, examples, tone, detail, or reasoning.\n"
            prompt += "\nUser-authored text: \n"
            prompt += post
            prompt += "\nLLM-generated text: \n"
            prompt += LLM_output 

            if debug:
                print(prompt)
            
            if model_name=="gemini-3-flash-preview":
                # USE GEMINI FREE TIER
                response = gemini.AskGoogleGemini(prompt, force=False)
            else:
                response = make_inference_call(prompt, model_name, force=False)
            
            if 'true' in response.lower():
                response = 'True'
            elif 'false' in response.lower():
                response = 'False'

            predicted_labels.append(response)

            if debug:
                print("LLM Label:", response)
                if groundtruth_df['annotator_1'][i] == 1: # Change to 'majority_vote' later
                    print("MY LABEL: True" )
                else:
                    print("MY LABEL: False")
                print()


        # Change predicted labels to 1 and 0
        for i, item in enumerate(predicted_labels):
            item = str(item).lower()
            predicted_labels[i] = item.replace(".", "")
            if item.lower() == 'true':
                predicted_labels[i] = 1
            else:
                predicted_labels[i] = 0

        models_predictions[model_name] = predicted_labels # Add each models predicitions to dictionary

        F1_Score_Recall_Precision(true_labels, predicted_labels)
        kappa = cohen_kappa_score(true_labels, predicted_labels)
        print("Cohen kappa score:", kappa)

    # agreement_matrix = compute_agreement_matrix(models_predictions)
    # print(agreement_matrix)
    # quit()

    models_predictions['jaccard_sim'] = jaccard_sim
    models_predictions['post_lengths'] = post_distances
    models_predictions['fre_difference'] = fre_differences

    # Predictions and rounds used from agentic deliberation with Gemini+LLama 
    models_predictions['agentic_predictions'] = gemini_and_llama_debate_df['agentic_predictions']
    models_predictions['rounds_used'] = gemini_and_llama_debate_df['rounds_used']

    # Gemini judge with Gemini+Llama independent agents 
    models_predictions['judge_final_label'] = gemini_judge_df['agentic_predictions']
    models_predictions['agent1_label'] = gemini_judge_df['agent1_label']
    models_predictions['agent2_label'] = gemini_judge_df['agent2_label']

    # quit()

    # df = pd.DataFrame(models_predictions)
    # df.to_csv("scores_for_ensemble_models.csv", index=False)
    # quit()

    #### Using results file ####
    df = pd.read_csv('Results/true_ensemble_models_scores.csv')
    # models_predictions = df.to_dict(orient="list")
    cosine_similarities = df['cosine_sim'].to_list()
    models_predictions['cosine_sim'] = cosine_similarities
    print(models_predictions.keys())

    # Run grid search over
    # logistic_results_df = run_logistic_grid_search(models_predictions, true_labels)
    # quit()

    selected_models = {}
    selected_models['gemini-3-flash-preview'] = models_predictions['gemini-3-flash-preview']
    selected_models['deepseek-ai/DeepSeek-R1'] = models_predictions['deepseek-ai/DeepSeek-R1']
    selected_models['jaccard_sim'] = models_predictions['jaccard_sim']
    selected_models['post_lengths'] = models_predictions['post_lengths']
    selected_models['agentic_predictions'] = models_predictions['agentic_predictions']
    selected_models['agent1_label'] = models_predictions['agent1_label']
    selected_models['agent2_label'] = models_predictions['agent2_label']

    # Get the best logistic regression parameters
    # results_df, best_params = logistic_regression_hyperparameter_tuning(true_labels, models_predictions=selected_predictions, n_splits=5)
    # print(best_params)
    # quit()

    # Run meta-classifier using the best parameters (found above)
    # y_true, y_pred = logistic_regression_classifier_cv(true_labels, models_predictions=selected_models, n_splits=5)
    # print_results("Logistic regression", y_true=y_true, y_pred=y_pred)
    # quit()
    ###########################################################

    # llm_names = [
    #     "gemini-2.5-flash",
    #     "Qwen/Qwen2.5-7B-Instruct",
    #     "deepseek-ai/DeepSeek-R1",
    #     "openai/gpt-oss-120b",
    #     "meta-llama/Llama-3.3-70B-Instruct"
    # ]

    # print("ABLATION: REMOVE ONE LLM, KEEP COSINE_SIM")

    # for removed_llm in llm_names:
    #     selected_predictions = {}

    #     for model_name in llm_names:
    #         if model_name != removed_llm:
    #             selected_predictions[model_name] = models_predictions[model_name]

    #     # always keep cosine similarity
    #     selected_predictions["cosine_sim"] = models_predictions["cosine_sim"]

    #     print("**"*40)
    #     print("Removed LLM:", removed_llm)
    #     print("Using:", list(selected_predictions.keys()))

    #     y_true_lr, y_pred_lr = logistic_regression_classifier_cv(
    #         selected_predictions,
    #         true_labels,
    #         n_splits=5
    #     )

    #     print("Logistic Regression Results:")
    #     F1_Score_Recall_Precision(y_true_lr, y_pred_lr)
    #     print(confusion_matrix(y_true_lr, y_pred_lr))

    #     kappa = cohen_kappa_score(y_true_lr, y_pred_lr)
    #     print("Cohen kappa score:", kappa)

    ###########################################################
    # Baseline comparisons    
    # majority_pred = majority_vote_classifier(models_predictions, true_labels)
    # print('Majority vote:')
    # F1_Score_Recall_Precision(true_labels, majority_pred)
    # print(confusion_matrix(true_labels, majority_pred))
    # quit()

    user_outputs = groundtruth_df['User Post'].to_list()
    LLM_outputs = groundtruth_df['LLM Output'].to_list()

    model = SentenceTransformer('all-MiniLM-L6-v2')

    user_embeddings = model.encode(user_outputs)
    LLM_embeddings = model.encode(LLM_outputs)

    X_embeddings = np.concatenate([
        user_embeddings,
        LLM_embeddings,
        np.abs(user_embeddings - LLM_embeddings),
        user_embeddings * LLM_embeddings
    ], axis=1)

    def print_fold_scores(scores):
        print("\nMean fold scores:")
        for metric in scores:
            print(metric, np.mean(scores[metric]))

        print("\nStd fold scores:")
        for metric in scores:
            print(metric, np.std(scores[metric]))

    y_true, y_pred, scores = logistic_regression_classifier_cv(true_labels, X=X_embeddings)
    print_results("Pooled Embedding Logistic Regression", y_true, y_pred)
    print_fold_scores(scores)

    y_true, y_pred, scores = svm_linear_classifier_cv(X_embeddings, true_labels)
    print_results("Pooled Embedding Linear SVM", y_true, y_pred)
    print_fold_scores(scores)

    y_true, y_pred, scores = random_forest_classifier(X_embeddings, true_labels)
    print_results("Pooled Embedding Random Forest", y_true, y_pred)
    print_fold_scores(scores)

    y_true, y_pred, scores = gradient_boosting_classifier(X_embeddings, true_labels)
    print_results("Pooled Embedding Gradient Boosting", y_true, y_pred)
    print_fold_scores(scores)

    y_true, y_pred, scores = xgboost_classifier(X_embeddings, true_labels)
    print_results("Pooled Embedding XGBoost", y_true, y_pred)
    print_fold_scores(scores)

if __name__=="__main__":
    main()