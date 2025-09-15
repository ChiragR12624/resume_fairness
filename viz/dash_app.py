# viz/dash_app.py
import os, pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fold_csv = os.path.join(BASE_DIR, "reports", "metrics_by_fold.csv")
df = pd.read_csv(fold_csv) if os.path.exists(fold_csv) else pd.DataFrame()

methods = df["Method"].unique().tolist() if not df.empty else ["Baseline"]
metrics = df["Metric"].unique().tolist() if not df.empty else ["Accuracy"]

app.layout = html.Div([
    html.H2("Fairness Dashboard (Dash)"),
    dcc.Dropdown(id="metric", options=[{"label":m,"value":m} for m in metrics], value=metrics[0]),
    dcc.Graph(id="box"),
])

@app.callback(Output("box","figure"), Input("metric","value"))
def update(metric):
    if df.empty:
        return px.scatter(title="No data found. Export reports/metrics_by_fold.csv from Day12.")
    fig = px.box(df[df["Metric"]==metric], x="Method", y="Value", points="all")
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
