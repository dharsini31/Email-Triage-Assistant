from flask import Flask, render_template, request
from agent.triage_agent import EmailTriageAgent

app = Flask(__name__)
agent = EmailTriageAgent()

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        email_text = request.form.get("email_text")

        if email_text:
            result = agent.process_email(email_text)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
