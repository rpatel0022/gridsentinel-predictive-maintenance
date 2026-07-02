# Deploying GridSentinel

Three surfaces, from zero-effort to full stack.

## 1. Live interactive dashboard — Streamlit Community Cloud (free, ~2 min)

The Streamlit app (`reports/app.py`) reads the committed real-data assets in
`reports/assets/`, so it deploys with **no model build and no secrets**.

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → select this repo, branch `main`, main file path `reports/app.py`.
3. **Deploy.** Dependencies install automatically from `requirements.txt`.

You'll get a public URL like `https://<your-app>.streamlit.app`. Every push to
`main` redeploys it.

## 2. Static results board — GitHub Pages (already live)

<https://rpatel0022.github.io/gridsentinel-predictive-maintenance/> — the Plotly
results board, served from the repo-root `index.html` (regenerate with `make pages`).

## 3. Live inference API — Render / Fly / a container host (optional)

The FastAPI service (`serving/`) needs a built model bundle, which is trained from
the real MetroPT-3 CSV (never committed). Two ways to host it:

- **Build at deploy time:** in the platform's build step, fetch the data and run
  `make artifact` to produce `models/`, then start `uvicorn serving.app:app`.
- **Bake into an image:** build the bundle locally, copy `models/` into the image
  via the provided `Dockerfile`, and deploy that image.

Locally, `make docker` brings up the full stack (API + Prometheus + Grafana) and
`make serve` runs just the API at <http://localhost:8000/docs>.
