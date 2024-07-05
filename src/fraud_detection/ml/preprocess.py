from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_pipeline() -> Pipeline:
    """Preprocess features data by
        - Encoded categorical fields
        - Scale data with Standard scaler

    Returns:
        Pipeline: Pipeline with data preprocessed
    """
    category_transformer = OneHotEncoder(drop='if_binary', sparse_output=False)
    numeric_transformer = make_pipeline(StandardScaler())
    
    column_transformer = make_column_transformer(
		(category_transformer, make_column_selector(dtype_include=[object, bool])),
		(numeric_transformer, make_column_selector(dtype_exclude=[object, bool])),
		remainder='passthrough'
	)

    return  make_pipeline(column_transformer)