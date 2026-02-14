from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QueryPlanner:

    def generate_plan(self, schema_summary, user_query):

        prompt = f"""
        You are a data science agent.
        Based on the dataset schema below:
        {schema_summary}

        User Query:
        {user_query}

        Generate a short analysis plan.
        Keep response concise.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content
