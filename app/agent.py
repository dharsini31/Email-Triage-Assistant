from app.schema import SchemaCompressor
from app.memory import HistoryManager
from app.eda import EDAEngine
from app.planner import QueryPlanner

class DataAnalysisAgent:

    def __init__(self):
        self.schema = SchemaCompressor()
        self.memory = HistoryManager()
        self.eda = EDAEngine()
        self.planner = QueryPlanner()

    def analyze(self, df, user_query):

        schema_summary = self.schema.compress(df)

        plan = self.planner.generate_plan(schema_summary, user_query)

        eda_results = self.eda.run_basic_eda(df)

        result = {
            "analysis_plan": plan,
            "schema_summary": schema_summary,
            "eda_results": eda_results,
            "history_context": self.memory.get_context()
        }

        self.memory.add_entry(user_query, result)

        return result
