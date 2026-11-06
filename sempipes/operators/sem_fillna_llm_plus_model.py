import copy
import io
import re
import tokenize
from collections.abc import Iterable
from typing import Any, Self

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from skrub import DataOp

from sempipes.code_generation.safe_exec import safe_exec
from sempipes.config import get_config
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.operators.operators import SemFillNAOperator

_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates code for the data imputation model. Answer only with the Python code."
)

# The generated code may only instantiate models, so SemPipes trains AutoGluon predictors itself with this time budget.
_AUTOGLUON_TIME_LIMIT_SECONDS = 60


class SemFillNALLLMPlusModel(SemFillNAOperator):
    def generate_imputation_estimator(self, data_op: DataOp, target_column: str, nl_prompt: str):
        return LearnedImputer(target_column, nl_prompt)


def _without_comments(code: str) -> str:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        return tokenize.untokenize(token for token in tokens if token.type != tokenize.COMMENT)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.sub(r"#.*", "", code)


def _as_strings(X: pd.DataFrame) -> pd.DataFrame:
    """Represent categorical feature values as plain strings, with a dedicated value for missing entries."""
    return pd.DataFrame(X).astype("string").fillna("<missing>").astype(object)


def _create_feature_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Standardise numeric and one-hot encode categorical feature columns. The target column is never part of X."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("as_strings", FunctionTransformer(_as_strings)),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def _is_autogluon_model(model: Any) -> bool:
    return type(model).__module__.startswith("autogluon")


def _has_own_column_preprocessing(model: Any) -> bool:
    """True for sklearn pipelines that select and encode the DataFrame columns themselves."""
    if isinstance(model, ColumnTransformer):
        return True
    if isinstance(model, Pipeline):
        return any(_has_own_column_preprocessing(step) for _, step in model.steps)
    return False


def _fresh_copy(model: Any) -> Any:
    try:
        return clone(model)
    except Exception:  # pylint: disable=broad-exception-caught
        return copy.deepcopy(model)


class _ImputerAsPredictor:
    """Turns an imputer (e.g. IterativeImputer, KNNImputer) into a predictor for the target column.

    The imputer sees the preprocessed features plus the target as last column (category codes for a categorical
    target). For prediction, the target column is all-missing and the imputer fills it in.
    """

    def __init__(self, imputer: Any, preprocessor: ColumnTransformer, is_numeric_target: bool):
        self.imputer = imputer
        self.preprocessor = preprocessor
        self.is_numeric_target = is_numeric_target
        self.categories_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        features = self.preprocessor.fit_transform(X)
        if self.is_numeric_target:
            target = pd.to_numeric(y).to_numpy(dtype=float)
        else:
            self.categories_ = np.array(sorted(pd.unique(y), key=str), dtype=object)
            target = pd.Categorical(y, categories=self.categories_).codes.astype(float)
        self.imputer.fit(np.column_stack([features, target]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        features = self.preprocessor.transform(X)
        missing_target = np.full(features.shape[0], np.nan)
        imputed = np.asarray(self.imputer.transform(np.column_stack([features, missing_target])))
        predictions = imputed[:, -1]
        if self.categories_ is None:
            return predictions
        codes = np.clip(np.rint(predictions), 0, len(self.categories_) - 1).astype(int)
        return self.categories_[codes]


class _AutoGluonAsPredictor:
    """Trains an AutoGluon TabularPredictor (instantiated by the generated code) on the raw DataFrame."""

    def __init__(self, predictor: Any):
        self.predictor = predictor

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        label = getattr(self.predictor, "label", y.name)
        self.predictor.fit(pd.concat([X, y.rename(label)], axis=1), time_limit=_AUTOGLUON_TIME_LIMIT_SECONDS)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.predictor.predict(X))


def _as_predictor(model: Any, X: pd.DataFrame, is_numeric_target: bool) -> Any:
    """Wrap a suggested model so that it can be trained with `fit(X, y)` and used via `predict(X)`."""
    if _is_autogluon_model(model):
        return _AutoGluonAsPredictor(model)
    if not hasattr(model, "fit"):
        raise TypeError(f"Suggested model {model} has no `fit` method")
    if hasattr(model, "predict"):
        if _has_own_column_preprocessing(model):
            return _fresh_copy(model)
        return Pipeline([("prep", _create_feature_preprocessor(X)), ("model", _fresh_copy(model))])
    if hasattr(model, "transform"):
        return _ImputerAsPredictor(_fresh_copy(model), _create_feature_preprocessor(X), is_numeric_target)
    raise TypeError(f"Suggested model {model} has neither a `predict` nor a `transform` method")


def _error(y_true: pd.Series, y_pred: Any, is_numeric_target: bool) -> float:
    """Validation error of a candidate (lower is better): RMSE for numeric targets, 1 - accuracy otherwise."""
    if is_numeric_target:
        return float(np.sqrt(mean_squared_error(y_true.to_numpy(dtype=float), np.asarray(y_pred, dtype=float))))
    return 1.0 - accuracy_score(y_true.astype(str).to_numpy(), pd.Series(np.asarray(y_pred)).astype(str).to_numpy())


class LearnedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, nl_prompt: str):
        self.target_column = target_column
        self.nl_prompt = nl_prompt

    @staticmethod
    def _build_prompt(
        target_column: str, target_column_type: str, candidate_columns: Iterable[str], nl_prompt: str
    ) -> str:
        return f"""
The data scientist wants to fill missing values in the column '{target_column}' of type '{target_column_type}'
in a dataframe. The dataframe has the following columns available: {candidate_columns}.

Choose which columns to use to fill the missing values. Special care: {nl_prompt}.

Code formatting:
```python
__chosen_columns = [<subset of `candidate_columns`>]
```end
"""

    @staticmethod
    def _build_model_prompt(
        target_column: str,
        target_column_type: str,
        selected_columns: Iterable[str],
        nl_prompt: str,
        is_numeric_target: bool,
    ) -> str:
        target_type = "numeric" if is_numeric_target else "categorical"
        try:
            from autogluon.tabular import __version__ as ag_version  # pylint: disable=import-outside-toplevel
        except ImportError:
            ag_version = "unknown"

        return f"""
Suggest five very different models for missing value imputation in column '{target_column}' (type: {target_column_type}, {target_type}).
Available columns: {selected_columns}. Special care: {nl_prompt}.

Use sklearn models (RandomForestRegressor, LinearRegression, etc.) or imputers (IterativeImputer) or AutoGluon (TabularPredictor).

CRITICAL 1: If you use AutoGluon's TabularPredictor, instantiate it with `label='{target_column}'`. SemPipes trains it on the raw DataFrame with a time limit of {_AUTOGLUON_TIME_LIMIT_SECONDS} seconds.

CRITICAL 2: Models must have `fit_transform` or `fit`/`predict` methods.

CRITICAL 3: Do NOT write any training code. Only instantiate models. The training and evaluation will be handled elsewhere.

Check that you use correct inputs for the models, e.g., pandas DataFrames vs numpy arrays.

Ensure that you can handle both numeric and categorical columns that contain text.

IMPORTANT 1: For sklearn models and imputers, the data has ALREADY been preprocessed before being passed to these models: numeric feature columns are standardised and categorical feature columns are one-hot encoded. Imputers additionally receive the target column as the last column (as integer category codes for a categorical target). You MUST NOT add any further preprocessing, feature selection, or column-based transformations.
IMPORTANT1b: Do NOT call `fit`, `fit_transform`, `predict`, or `transform`. Only instantiate model objects. Do NOT create dummy dataframes or use `df` in the code.
Versions: sklearn {sklearn.__version__}, pandas {pd.__version__}, autogluon {ag_version}.
You are NOT allowed to use `tempfile` library.

IMPORTANT 2: Always import IterativeImputer via `from sklearn.experimental import enable_iterative_imputer` first.
```python
from sklearn.experimental import enable_iterative_imputer # Always import IterativeImputer via this first
from sklearn.impute import IterativeImputer

model1 = ...  # Missing data imputation model
model2 = ...  # Missing data imputation model
model3 = ...  # Missing data imputation model
model4 = ...  # Missing data imputation model
model5 = ...  # Missing data imputation model
suggested_models_and_imputers = [model1, model2, model3, model4, model5]
```end

Codeblock:
"""

    def _generate_candidate_models(self, target_column_type: str, is_numeric_target: bool) -> list[Any]:
        """Ask the LLM for candidate models (retrying with error feedback). Returns an empty list on failure."""
        prompt = self._build_model_prompt(
            self.target_column, target_column_type, self.feature_columns_, self.nl_prompt, is_numeric_target
        )
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        last_exception = None

        for attempt in range(1, get_config().max_retries_for_code_gen + 1):
            code = ""
            try:
                code = generate_python_code_from_messages(messages)
                print(code)

                forbidden = [
                    r"\bfit_transform\(",
                    r"\bfit\(",
                    r"\bpredict\(",
                    r"\btransform\(",
                    r"\bdf\b",
                ]
                if any(re.search(pattern, _without_comments(code)) for pattern in forbidden):
                    raise ValueError(
                        "Generated code must only instantiate model objects (no fit/predict/transform or df usage)."
                    )

                return list(safe_exec(code, "suggested_models_and_imputers"))
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_exception = e
                print(f"Attempt {attempt} failed: {e}")
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"Code execution failed: {type(e).__name__}: {e}\nCode: ```python{code}```\nPlease generate corrected code:\n```python\n",
                    },
                ]

        print(
            f"sem_fillna code generation failed after {get_config().max_retries_for_code_gen} attempts "
            f"({type(last_exception).__name__}: {last_exception}); using the fallback imputer"
        )
        return []

    def fit(self, df: pd.DataFrame, y=None) -> Self:  # pylint: disable=unused-argument
        print(f"--- sempipes.sem_fillna('{self.target_column}', '{self.nl_prompt}')")

        target_column_type = str(df[self.target_column].dtype)
        self.feature_columns_ = [c for c in df.columns if c != self.target_column]

        X = df[self.feature_columns_]
        target = df[self.target_column]
        is_numeric_target = pd.api.types.is_numeric_dtype(target)
        known_mask = target.notna()
        X_known, y_known = X[known_mask], target[known_mask]

        suggested_models = self._generate_candidate_models(target_column_type, is_numeric_target)

        # Score every candidate on a validation split of the rows where the target is known
        candidates: list[Any] = []
        errors: list[float] = []
        if suggested_models and len(y_known) >= 10:
            X_train, X_val, y_train, y_val = train_test_split(X_known, y_known, test_size=0.2, random_state=42)
            for model_index, model in enumerate(suggested_models, 1):
                try:
                    candidate = _as_predictor(model, X, is_numeric_target)
                    candidate.fit(X_train, y_train)
                    error = _error(y_val, candidate.predict(X_val), is_numeric_target)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Error evaluating model {model_index}: {e}")
                    candidate, error = None, float("inf")
                print(f"Validation error of model {model_index}: {error}")
                candidates.append(candidate)
                errors.append(error)

        # Refit the best candidate on all rows with a known target. Candidates that failed evaluation are skipped.
        for index in [i for i in np.argsort(errors) if np.isfinite(errors[i])]:
            print(f"Selected model {index + 1} with validation error {errors[index]}, fitting on {len(y_known)} rows")
            try:
                if isinstance(candidates[index], _AutoGluonAsPredictor):
                    # AutoGluon predictors cannot be refitted, so we keep the one trained during evaluation
                    self.imputation_model_ = candidates[index]
                else:
                    self.imputation_model_ = _as_predictor(suggested_models[index], X, is_numeric_target).fit(
                        X_known, y_known
                    )
                return self
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Keep the copy trained on the validation split (80% of the known rows) instead of discarding it
                print(f"Refitting model {index + 1} on all rows failed ({e}), using the model trained for validation")
                self.imputation_model_ = candidates[index]
                return self

        fallback = DummyRegressor(strategy="median") if is_numeric_target else DummyClassifier(strategy="most_frequent")
        print(f"No suggested model could be fitted, falling back to {fallback}")
        self.imputation_model_ = _as_predictor(fallback, X, is_numeric_target).fit(X_known, y_known)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "imputation_model_")
        missing_mask = df[self.target_column].isna()

        if not missing_mask.any():
            return df

        print(f"\t> Imputing {missing_mask.sum()} values...")
        df = df.copy()
        predictions = self.imputation_model_.predict(df.loc[missing_mask, self.feature_columns_])
        df.loc[missing_mask, self.target_column] = np.asarray(predictions)
        return df
