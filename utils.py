import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List
from types import SimpleNamespace
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    get_scorer,
)
import warnings
warnings.filterwarnings('ignore')


class CNC():
    def __init__(self, args: SimpleNamespace):
        self.args = args
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.model_map = {
            "Logistic Regression": LogisticRegression(random_state=self.args.seed),
            "Gaussian Naive Bayes":  GaussianNB(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=self.args.seed),
            "Random Forest": RandomForestClassifier(random_state=self.args.seed),
            "eXtreme Gradient Boosting": XGBClassifier(random_state=self.args.seed),
        }
        self.sampler_map = {
            "RandomOverSampler": RandomOverSampler(random_state=self.args.seed, sampling_strategy=self.args.sampling_strategy),
            "SMOTE": SMOTE(random_state=self.args.seed, sampling_strategy=self.args.sampling_strategy),
            "ADASYN": ADASYN(random_state=self.args.seed, sampling_strategy=self.args.sampling_strategy),
            "BorderlineSMOTE": BorderlineSMOTE(random_state=self.args.seed, sampling_strategy=self.args.sampling_strategy),
            "SVMSMOTE": SVMSMOTE(random_state=self.args.seed, sampling_strategy=self.args.sampling_strategy),
        }

    def pre_process(self):
        self._load_data()
        fig = self._features_selection()
        self._create_rolling_features()
        self._split_data()
        return fig

    def train(self):
        self.models = []
        self.tuned_models = []
        for i in range(self.args.future_steps):
            model = self.model_map[self.args.model]
            x, y = self.x_train, self.y_train[i]
            print(f"🔧 Training step {i+1}: {len(x)} samples, {len(np.unique(y))} classes")
            
            x, y, figs = self._sampling(x, y)
            print(f"🔧 After sampling: {len(x)} samples, {len(np.unique(y))} classes")
            
            if model is None:
                raise ValueError(f"Unknown model type: {self.args.model}")
            
            # Check if we have enough samples for cross-validation
            if len(x) < 5:
                print(f"⚠️ Warning: Only {len(x)} samples available for training. This may cause issues with cross-validation.")
            
            model.fit(x, y)
            self.models.append(model)
            # Use custom threshold or automatic tuning
            if getattr(self.args, 'use_custom_threshold', False) and getattr(self.args, 'custom_threshold', None) is not None:
                # Use custom threshold
                from sklearn.base import BaseEstimator, ClassifierMixin
                
                class CustomThresholdClassifier(BaseEstimator, ClassifierMixin):
                    def __init__(self, estimator, threshold):
                        self.estimator = estimator
                        self.threshold = threshold
                    
                    def fit(self, X, y):
                        self.estimator.fit(X, y)
                        return self
                    
                    def predict(self, X):
                        proba = self.estimator.predict_proba(X)[:, 1]
                        return (proba >= self.threshold).astype(int)
                    
                    def predict_proba(self, X):
                        return self.estimator.predict_proba(X)
                
                tuned_model = CustomThresholdClassifier(model, self.args.custom_threshold)
                print(f"🎛️ Using custom threshold: {self.args.custom_threshold}")
            else:
                # Use automatic tuning with selected strategy
                tuning_strategy = getattr(self.args, 'tuning_strategy', 'f1')
                tuned_model = TunedThresholdClassifierCV(
                    estimator=model,
                    scoring=tuning_strategy,
                    store_cv_results=True,
                    random_state=self.args.seed,
                )
                print(f"🎯 Using automatic tuning with metric: {tuning_strategy}")
            
            tuned_model.fit(x, y)
            self.tuned_models.append(tuned_model)
        return figs

    def evaluate(self):
        self.vanilla_preds = []
        self.tuned_preds = []
        self.risk_assessments = []
        figs = []
        balanced_accuracy_scorer = get_scorer("balanced_accuracy")
        
        for i in range(self.args.future_steps):
            
            vanilla_pred = self.models[i].predict(self.x_test)
            vanilla_report = classification_report(self.y_test[i], vanilla_pred, output_dict=True, target_names=['Non-Anomaly', 'Anomaly'])
            vanilla_report["balanced_accuracy"] = balanced_accuracy_scorer(self.models[i], self.x_test, self.y_test[i])

            tuned_pred = self.tuned_models[i].predict(self.x_test)
            tuned_report = classification_report(self.y_test[i], tuned_pred, output_dict=True, target_names=['Non-Anomaly', 'Anomaly'])
            tuned_report["balanced_accuracy"] = balanced_accuracy_scorer(self.tuned_models[i], self.x_test, self.y_test[i])

            self.vanilla_preds.append(vanilla_pred)
            self.tuned_preds.append(tuned_pred)
            
            # Risk assessment
            risk_assessment = self._assess_risk(i)
            self.risk_assessments.append(risk_assessment)
            
            figs.append(self._visualize_classification_report(i + 1, vanilla_report, tuned_report))
            figs.append(self._plot_roc_pr_curves(i))
            figs.append(self._plot_confusion_matrix(i))
            
            # Add risk assessment visualization
            if hasattr(self.args, 'risk_mode'):
                risk_fig = self._visualize_risk_assessment(i, risk_assessment)
                figs.append(risk_fig)
        
        return figs

    def _load_data(self):
        data = self.args.data
        
        # Process time column if it exists
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time']).astype('int64') // 10**9
        
        # Use detected target column
        target_column = getattr(self.args, 'target_column', 'Anomaly')
        
        # Drop target column and keep only numeric columns
        self.x = data.drop(columns=[target_column])
        
        # Filter to only numeric columns for ML processing
        numeric_columns = self.x.select_dtypes(include=[np.number]).columns
        self.x = self.x[numeric_columns]
        
        # Print info about columns being used
        print(f"Using {len(numeric_columns)} numeric features: {list(numeric_columns)}")
        print(f"Target column: {target_column}")
        
        self.y = data[target_column]

    def _features_selection(self):
        
        correlation_matrix = self.x.corr()

        high_corr_pairs = [
            (correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
            for i in range(len(correlation_matrix.columns))
            for j in range(i + 1, len(correlation_matrix.columns))
            if abs(correlation_matrix.iloc[i, j]) > self.args.corr_upper_threshold or abs(correlation_matrix.iloc[i, j]) < self.args.corr_lower_threshold
        ]
        
        features_to_remove = {feature2 for _, feature2, _ in high_corr_pairs}
        
        remaining_features = [feature for feature in self.x.columns if feature not in features_to_remove]
        self.x = self.x[remaining_features]
        filtered_correlation_matrix = self.x.corr()

        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=400)
        fig.suptitle('Features Selection Before and After', fontsize=22, fontweight='bold')

        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, ax=ax1, annot_kws={"size": 10})
        ax1.set_title('Feature Correlation Matrix (Before)', fontsize=18, fontweight='bold')
        ax1.tick_params(axis='x', rotation=90, labelsize=12)
        ax1.tick_params(axis='y', rotation=0, labelsize=12)

        sns.heatmap(filtered_correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, ax=ax2, annot_kws={"size": 10})
        ax2.set_title('Feature Correlation Matrix (After)', fontsize=18, fontweight='bold')
        ax2.tick_params(axis='x', rotation=90, labelsize=12)
        ax2.tick_params(axis='y', rotation=0, labelsize=12)

        plt.tight_layout()
        return fig

    def _create_rolling_features(self):
        x, y = [], []
        for i in range(len(self.y) - self.args.window_size - self.args.future_steps + 1):
            features = self.x.iloc[i:i + self.args.window_size].values.flatten()
            labels = self.y[i + self.args.window_size:i + self.args.window_size + self.args.future_steps].values
            x.append(features)
            y.append(labels)
        self.x = np.array(x)
        self.y = np.array(y)

    def _split_data(self):
        y_list: List[np.ndarray] = [self.y[:, i] for i in range(self.args.future_steps)]
        print(f"📊 Total samples before split: {len(self.x)}")
        print(f"📊 Test size requested: {self.args.test_size}")
        
        # Ensure test_size doesn't leave too few training samples
        if self.args.test_size >= len(self.x):
            test_size = max(1, len(self.x) - 10)  # Leave at least 10 samples for training
            print(f"⚠️ Test size too large, using {test_size} instead")
        else:
            test_size = self.args.test_size
            
        self.x_train, self.x_test, *y_splits = train_test_split(self.x, *y_list, test_size=test_size, random_state=self.args.seed)
        self.y_train = [y_splits[i] for i in range(0, len(y_splits), 2)]
        self.y_test = [y_splits[i] for i in range(1, len(y_splits), 2)]
        
        print(f"📊 Training samples: {len(self.x_train)}")
        print(f"📊 Test samples: {len(self.x_test)}")

    def _sampling(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.args.sampler != "None":
            original_counts = np.bincount(y)
            original_counts = np.bincount(y)
            majority_class = np.argmax(original_counts)
            self.args.sampling_strategy[majority_class] = max(self.args.sampling_strategy[majority_class], original_counts[majority_class])
            sampler = self.sampler_map.get(self.args.sampler)
            x_res, y_res = sampler.fit_resample(x, y)
            figs = self._visualize_sampling(x, y, x_res, y_res)
        else:
            x_res, y_res, figs = x, y, None
        return x_res, y_res, figs

    def _visualize_sampling(self, x: np.ndarray, y: np.ndarray, x_res: np.ndarray, y_res: np.ndarray):
        fig_distribute, ax_distribute = plt.subplots(1, 2, figsize=(16, 9), dpi=400)
        label_names = ['Non-Anomaly', 'Anomaly']
        colors = ['#ff9999', '#66b3ff']
        counts = np.bincount(y)
        wedges_before, _, _ = ax_distribute[0].pie(
            counts, 
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*sum(counts))})', 
            startangle=90, 
            colors=colors, 
            textprops={'fontsize': 14}
        )
        ax_distribute[0].set_title('Before Sampling', fontsize=16)
        counts_resampled = np.bincount(y_res)
        ax_distribute[1].pie(
            counts_resampled,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*sum(counts_resampled))})',
            startangle=90,
            colors=colors, 
            textprops={'fontsize': 14}
        )
        ax_distribute[1].set_title('After Sampling', fontsize=16)

        fig_distribute.legend(wedges_before, label_names, loc='lower center', fontsize=14, title='Classes', ncol=2)
        plt.suptitle(f'Impact of {self.args.sampler} on Class Distribution', fontsize=20, y=0.95)
        plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(x)
        X_res_pca = pca.transform(x_res)

        fig_pca, ax_pca = plt.subplots(1, 2, figsize=(16, 9), dpi=400)
        
        ax_pca[0].scatter(
            X_pca[:, 0], X_pca[:, 1], c='blue', alpha=1, edgecolor='w', s=60, marker='o', label='Original'
        )
        
        ax_pca[0].set_title('Before Sampling', fontsize=16, color='black')
        ax_pca[0].set_xlabel('PCA Component 1', fontsize=14)
        ax_pca[0].set_ylabel('PCA Component 2', fontsize=14)
        
        ax_pca[1].scatter(
            X_pca[:, 0], X_pca[:, 1], c='blue', alpha=1, edgecolor='w', s=60, marker='o', label='Original'
        )
        ax_pca[1].scatter(
            X_res_pca[:, 0], X_res_pca[:, 1], c='red', alpha=0.8, edgecolor='k', s=30, marker='x', label='Resampled'
        )
        ax_pca[1].set_title('After Sampling', fontsize=16, color='black')
        ax_pca[1].set_xlabel('PCA Component 1', fontsize=14)
        ax_pca[1].set_ylabel('PCA Component 2', fontsize=14)
        
        for ax in ax_pca:
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#f5f5f5')
        
        handles, labels = ax_pca[1].get_legend_handles_labels()
        fig_pca.legend(handles, labels, loc='upper right', fontsize=14, ncol=2)
        
        plt.tight_layout(pad=2.0)
        fig_pca.suptitle(f'PCA Visualization of Original and {self.args.sampler} Data', fontsize=20, y=1.05)
        
        return [fig_distribute, fig_pca]

    def _visualize_classification_report(self, step: int, vanilla_report: dict, tuned_report: dict):
        classes = list(vanilla_report.keys())
        fig, ax = plt.subplots(figsize=(20, 8), dpi=400)
        width = 0.35

        vanilla_scores = []
        tuned_scores = []
        labels = []
        
        for cls in classes:
            if cls in ['accuracy', 'balanced_accuracy']:
                vanilla_scores.append(vanilla_report[cls])
                tuned_scores.append(tuned_report[cls])
                labels.append('Accuracy' if cls == 'accuracy' else 'Balanced Accuracy')
            elif cls in ['Non-Anomaly', 'Anomaly']:
                for metric in ['precision', 'recall', 'f1-score']:
                    vanilla_scores.append(vanilla_report[cls][metric])
                    tuned_scores.append(tuned_report[cls][metric])
                    labels.append(f'{cls} - {metric}')
        
        x = np.arange(len(labels))
        
        bars_vanilla = ax.bar(x - width/2, vanilla_scores, width, label='Before', color='skyblue', edgecolor='black')
        bars_tuned = ax.bar(x + width/2, tuned_scores, width, label='After', color='orange', edgecolor='black')
        
        for bar in bars_vanilla:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
        for bar in bars_tuned:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Scores', fontsize=12)
        ax.set_title(f"Comparison of Classifier Before and After Post-Tuning the Decision Threshold - Future step {step}", fontsize=18, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
        ax.legend(loc='best', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return fig

    def _plot_roc_pr_curves(self, i: int):
        pos_label, neg_label = True, False
        model_name = self.args.model

        def fpr_score(y, y_pred, neg_label, pos_label):
            cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
            tn, fp, _, _ = cm.ravel()
            tnr = tn / (tn + fp)
            return 1 - tnr

        tpr_score = recall_score
        scoring = {
            "precision": make_scorer(precision_score, pos_label=pos_label),
            "recall": make_scorer(recall_score, pos_label=pos_label),
            "fpr": make_scorer(fpr_score, neg_label=neg_label, pos_label=pos_label),
            "tpr": make_scorer(tpr_score, pos_label=pos_label),
        }

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(21, 7), dpi=400)
        linestyles = ("dashed", "dotted")
        markerstyles = ("o", ">")
        colors = ("tab:blue", "tab:orange")
        names = (f"Vanilla {model_name}", f"Tuned {model_name}")

        for idx, (est, linestyle, marker, color, name) in enumerate(
        zip((self.models[i], self.tuned_models[i]), linestyles, markerstyles, colors, names)
        ):
            decision_threshold = getattr(est, "best_threshold_", 0.5)
    
            pr_display = PrecisionRecallDisplay.from_estimator(
                est,
                self.x_test,
                self.y_test[i],
                pos_label=pos_label,
                linestyle=linestyle,
                color=color,
                ax=axs[0],
                name=name,
            )
            axs[0].plot(
                scoring["recall"](est, self.x_test, self.y_test[i]),
                scoring["precision"](est, self.x_test, self.y_test[i]),
                marker,
                markersize=10,
                color=color,
                label=f"Cut-off point at probability of {decision_threshold:.2f}",
            )

            roc_display = RocCurveDisplay.from_estimator(
                est,
                self.x_test,
                self.y_test[i],
                pos_label=pos_label,
                linestyle=linestyle,
                color=color,
                ax=axs[1],
                name=name,
                plot_chance_level=idx == 1,
            )
            axs[1].plot(
                scoring["fpr"](est, self.x_test, self.y_test[i]),
                scoring["tpr"](est, self.x_test, self.y_test[i]),
                marker,
                markersize=10,
                color=color,
                label=f"Cut-off point at probability of {decision_threshold:.2f}",
            )

        axs[0].fill_between(pr_display.recall, pr_display.precision, step='post', alpha=0.2, color="b")
        axs[0].set_title("Precision-Recall curve", fontsize=14)
        axs[0].legend(loc="best")
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].set_xlabel("Recall", fontsize=12)
        axs[0].set_ylabel("Precision", fontsize=12)

        axs[1].fill_between(roc_display.fpr, roc_display.tpr, alpha=0.2, color="b")
        axs[1].set_title("ROC Curve", fontsize=14)
        axs[1].legend(loc="best")
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].set_xlabel("False Positive Rate", fontsize=12)
        axs[1].set_ylabel("True Positive Rate", fontsize=12)

        axs[2].plot(
            self.tuned_models[i].cv_results_["thresholds"],
            self.tuned_models[i].cv_results_["scores"],
            color="tab:orange",
            linestyle='-',
        )
        axs[2].plot(
            self.tuned_models[i].best_threshold_,
            self.tuned_models[i].best_score_,
            "o",
            markersize=10,
            color="tab:orange",
            label=f"Optimal cut-off point for the Balanced Accuracy ({self.tuned_models[i].best_threshold_:.2f})",
        )
        axs[2].legend(loc="best", fontsize=12)
        axs[2].set_xlabel("Decision Threshold (Probability)", fontsize=12)
        axs[2].set_ylabel("Objective Score (Balanced Accuracy)", fontsize=12)
        axs[2].set_title("Objective score as a function of the decision threshold", fontsize=14)
        axs[2].grid(True, linestyle='--', alpha=0.7)
        
        fig.suptitle(f"Comparison of the cut-off point for the vanilla and tuned {model_name} - Future step {i+1}", fontsize=20)
        plt.tight_layout()
        return fig

    def _plot_confusion_matrix(self, i: int):
        vanilla_cm = confusion_matrix(self.y_test[i], self.vanilla_preds[i])
        tuned_cm = confusion_matrix(self.y_test[i], self.tuned_preds[i])
        
        fig, ax = plt.subplots(1, 2, figsize=(20, 9), dpi=400)
        
        sns.heatmap(vanilla_cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                xticklabels=['Non-Anomaly', 'Anomaly'], yticklabels=['Non-Anomaly', 'Anomaly'],
                annot_kws={"size": 14}, linewidths=1, linecolor='black', ax=ax[0])

        ax[0].set_xlabel('Predicted', fontsize=16, labelpad=20)
        ax[0].set_ylabel('Actual', fontsize=16, labelpad=20)
        ax[0].set_title(f'Before', fontsize=18, pad=20)
        ax[0].tick_params(axis='x', labelsize=14)
        ax[0].tick_params(axis='y', labelsize=14)

        sns.heatmap(tuned_cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                    xticklabels=['Non-Anomaly', 'Anomaly'], yticklabels=['Non-Anomaly', 'Anomaly'],
                    annot_kws={"size": 14}, linewidths=1, linecolor='black', ax=ax[1])

        ax[1].set_xlabel('Predicted', fontsize=16, labelpad=20)
        ax[1].set_ylabel('Actual', fontsize=16, labelpad=20)
        ax[1].set_title(f'After', fontsize=18, pad=20)
        ax[1].tick_params(axis='x', labelsize=14)
        ax[1].tick_params(axis='y', labelsize=14)

        fig.suptitle(f'Confusion Matrix of the Before and After Post-Tuning the Decision Threshold - Future step {i+1}', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _assess_risk(self, step_idx):
        """Assess risk and provide maintenance recommendations"""
        model = self.tuned_models[step_idx]
        
        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(self.x_test)[:, 1]  # Probability of anomaly
        else:
            probabilities = model.predict(self.x_test)
        
        # Risk assessment based on mode
        risk_assessment = {
            'high_risk_count': np.sum(probabilities > self.args.risk_threshold),
            'total_samples': len(probabilities),
            'risk_percentage': np.mean(probabilities > self.args.risk_threshold) * 100,
            'avg_risk_score': np.mean(probabilities),
            'max_risk_score': np.max(probabilities),
            'risk_distribution': probabilities,
            'recommendations': []
        }
        
        # Generate recommendations based on risk mode
        if self.args.risk_mode == 'High Risk Alert':
            if risk_assessment['risk_percentage'] > 20:
                risk_assessment['recommendations'].append("🚨 IMMEDIATE ACTION REQUIRED: High failure risk detected")
            elif risk_assessment['risk_percentage'] > 10:
                risk_assessment['recommendations'].append("⚠️ WARNING: Elevated failure risk - schedule maintenance")
            else:
                risk_assessment['recommendations'].append("✅ Normal operation - continue monitoring")
                
        elif self.args.risk_mode == 'Maintenance Planning':
            if risk_assessment['risk_percentage'] > 15:
                risk_assessment['recommendations'].append("🔧 Schedule preventive maintenance within 24 hours")
            elif risk_assessment['risk_percentage'] > 8:
                risk_assessment['recommendations'].append("📅 Plan maintenance within 1 week")
            else:
                risk_assessment['recommendations'].append("📊 Continue routine monitoring")
                
        elif self.args.risk_mode == 'Cost-Benefit Analysis':
            # Calculate expected costs
            expected_failure_cost = risk_assessment['avg_risk_score'] * self.args.failure_cost
            preventive_cost = self.args.maintenance_cost
            
            if expected_failure_cost > preventive_cost * 1.5:
                risk_assessment['recommendations'].append(f"💰 RECOMMENDED: Preventive maintenance saves ${expected_failure_cost - preventive_cost:.0f}")
            elif expected_failure_cost > preventive_cost:
                risk_assessment['recommendations'].append(f"⚖️ CONSIDER: Preventive maintenance saves ${expected_failure_cost - preventive_cost:.0f}")
            else:
                risk_assessment['recommendations'].append("✅ Cost-effective to continue operation")
        
        return risk_assessment

    def _visualize_risk_assessment(self, step_idx, risk_assessment):
        """Create risk assessment visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Risk Assessment Dashboard - Future Step {step_idx + 1}', fontsize=16, fontweight='bold')
        
        # 1. Risk Distribution Histogram
        axes[0, 0].hist(risk_assessment['risk_distribution'], bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].axvline(self.args.risk_threshold, color='orange', linestyle='--', linewidth=2, label=f'Alert Threshold ({self.args.risk_threshold})')
        axes[0, 0].set_xlabel('Risk Score (Probability of Failure)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Risk Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Risk Summary Pie Chart
        risk_labels = ['Low Risk', 'High Risk']
        risk_sizes = [100 - risk_assessment['risk_percentage'], risk_assessment['risk_percentage']]
        risk_colors = ['lightgreen', 'red']
        
        axes[0, 1].pie(risk_sizes, labels=risk_labels, colors=risk_colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Risk Level Distribution')
        
        # 3. Key Metrics Bar Chart
        metrics = ['Avg Risk', 'Max Risk', 'High Risk %']
        values = [risk_assessment['avg_risk_score'], risk_assessment['max_risk_score'], risk_assessment['risk_percentage']/100]
        
        bars = axes[1, 0].bar(metrics, values, color=['blue', 'red', 'orange'])
        axes[1, 0].set_ylabel('Risk Score')
        axes[1, 0].set_title('Key Risk Metrics')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Recommendations Text
        axes[1, 1].axis('off')
        recommendation_text = f"""
Risk Assessment Summary:
• High Risk Samples: {risk_assessment['high_risk_count']}/{risk_assessment['total_samples']}
• Risk Percentage: {risk_assessment['risk_percentage']:.1f}%
• Average Risk Score: {risk_assessment['avg_risk_score']:.3f}
• Maximum Risk Score: {risk_assessment['max_risk_score']:.3f}

Recommendations:
"""
        for rec in risk_assessment['recommendations']:
            recommendation_text += f"• {rec}\n"
        
        axes[1, 1].text(0.05, 0.95, recommendation_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        return fig