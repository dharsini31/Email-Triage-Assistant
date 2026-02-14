import pandas as pd

class EDAEngine:

    def run_basic_eda(self, df: pd.DataFrame) -> dict:
        numerical_df = df.select_dtypes(include='number')

        eda_results = {
            "shape": df.shape,
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "descriptive_stats": numerical_df.describe().to_dict() if not numerical_df.empty else {},
            "correlations": numerical_df.corr().to_dict() if not numerical_df.empty else {}
        }

        return eda_results
