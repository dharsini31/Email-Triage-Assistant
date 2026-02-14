import pandas as pd

class SchemaCompressor:

    def compress(self, df: pd.DataFrame) -> dict:
        schema_summary = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": []
        }

        for col in df.columns:
            column_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique()),
                "sample_values": df[col].dropna().unique()[:3].tolist()
            }

            schema_summary["columns"].append(column_info)

        return schema_summary
