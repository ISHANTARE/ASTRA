# Deployment & DevOps Strategy

## 1. Infrastructure Overview
Targeting free-tier scalable services suitable for an academic/research project.

- **Frontend Hosting:** Vercel
  - Automatically builds and deploys the React/Vite/Three.js application.
  - Global CDN provides fast loading for the visualization assets.
- **Backend Hosting:** Render
  - Deploys the Python FastAPI application via Docker or native Python execution.
  - Handles the raw computational load of SGP4 propagation and matrix operations via `numpy`.
- **Database / Auth:** Supabase (PostgreSQL)
  - Stores the parsed CelesTrak TLE catalog.
  - Allows the backend to quickly execute SQL-based filtering (Stage 1 & 2 of the multi-stage filter) before passing data to Python.

## 2. Automated Pipeline
- Daily Cron Job (Github Actions or Render Background Worker) to fetch fresh TLEs from `celestrak.org`.
- Database update overwrites outdated TLE entries.

## 3. Scaling Considerations
- If prediction computations run into standard 30-60s timeout limits on Render free tier, the backend should implement a background task queue (e.g., Celery + Redis, or async background tasks returning a Job ID) and the frontend should poll for completion.
