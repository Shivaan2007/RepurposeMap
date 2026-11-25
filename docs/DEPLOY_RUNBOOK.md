# Deploy runbook: Render (API) and Vercel (frontend)

Needs your own accounts, so nothing here has been run from this repo's tooling.
Config already in the repo: `render.yaml`, `web/app/vercel.json`,
`web/app/.env.example`. No database, Redis, API key or secret is needed.

The two services need each other's URL, so the order is: API, frontend, then CORS.

## 0. Before deploying: the serving data must be committed

The API reads `data/serving/*.json` from the repo checkout. Run the post-checkpoint
`export` step, commit and push `data/serving/` and `main` first. A deploy from a
commit without the bundles is a valid but empty demo (`not_available` for every disease).

## 1. Backend on Render

1. Render dashboard, New, Blueprint, pick this repo and branch `main`. It reads `render.yaml` (service `repurposemap-api`, free plan).
2. Build command `pip install -e ".[api]"`, start command `uvicorn repurposemap.api.app:app --host 0.0.0.0 --port $PORT`, health check `/health`. These come from the file; do not edit them.
3. When asked for `REPURPOSEMAP_CORS_ORIGINS` (the only variable), leave it empty for now and apply.
4. Wait for the first deploy. Note the URL, e.g. `https://repurposemap-api.onrender.com` (the exact name may differ).
5. Check: `curl -s <API>/health` and `curl -s <API>/api/v1/system/info`.

## 2. Frontend on Vercel

1. Vercel, Add New, Project, import this repo.
2. **Root Directory: `web/app`.** Framework preset: Vite. `vercel.json` sets build `npm install && npm run build`, output `dist`, and the SPA rewrite.
3. The app depends on `../client` (`file:../client`). If the build cannot find it, in Project Settings, General, turn on "Include source files outside of the Root Directory in the Build Step" and redeploy.
4. Environment variable (build-time, so changing it needs a redeploy): `VITE_API_BASE_URL` = the Render URL from step 1.4, no trailing slash.
5. Deploy. Note the URL, e.g. `https://repurposemap.vercel.app`.

## 3. Close the loop (CORS)

Render, the service, Environment: set `REPURPOSEMAP_CORS_ORIGINS` to the exact Vercel origin (scheme and host, no path or trailing slash). A comma-separated list is allowed (add the preview URL if you use one). Never `*`. Save; the service restarts.

## 4. Verify

```bash
./scripts/smoke_test.sh https://<your-render-url>
```
Then by hand: open the Vercel URL, confirm the status panel loads with no console error, `GET <API>/api/v1/rank/diseases` lists the exported diseases, a ranked disease returns `serving_mode: "precomputed"`, an unexported one returns `model_status: "not_available"`, and an empty disease name shows an honest error. Free Render instances sleep when idle, so the first request after a pause can be slow: retry before calling it broken.

## 5. Record it

Put the two URLs into the README "Live demo" block (`docs/RELEASE_TEXT_TEMPLATES.md` §B) and re-capture the screenshot from the live site (`docs/DEMO_MATERIALS.md`).
