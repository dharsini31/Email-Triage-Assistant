import pandas as pd
import json
from app.agent import DataAnalysisAgent

if __name__ == "__main__":

    df = pd.read_csv("sample.csv")

    agent = DataAnalysisAgent()

    output = agent.analyze(df, "Perform full exploratory data analysis")

    print(json.dumps(output, indent=2))
