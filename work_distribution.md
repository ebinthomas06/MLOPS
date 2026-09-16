## MLOPS

## AutoMLOps — Complete Work Distribution Guide

## A Self-Hosted MLOps Platform with AI-Powered Version Diagnostics

## What are we building? (Read this first, everyone)

We are building a platform that an ML engineer can run on their own server.

## The ML engineer:

- ý. Opens our web UI

- 2. Uploads their training script ( .py) and dataset ( .csv)

- 3. Sets success criteria (e.g. "accuracy must be above 85%")

- 4. Clicks submit

## Our platform then:

- ý. Runs their training script inside a safe isolated environment

- 2. Checks if the trained model meets their criteria

- 3. If it passes → deploys it and gives back a live prediction API link

- 4. If it fails → tells them why and stops

- 5. When they upload a new version later → compares old vs new, keeps the better one live, and uses AI to explain what changed and why

## Project Folder Structure (everyone should know this)


## Phase 1 — Everyone sets up locally (Week 1, all 5 people, parallel)

Before writing any code, everyone must have the same environment.


## Steps for every team member:

- ý. Install Python 3.10+

- 2. Install Docker Desktop

- 3. Clone the project repo from GitHub

- 4. Install Python dependencies: pip install fastapi uvicorn mlflow evidently pandas scikit-learn python-multipart

- 5. Install Node.js and run npm install inside the frontend/ folder

- 6. Make sure Docker is running — type docker ps in terminal, it should not give an error

At the end of Week 1: Everyone can run the project locally without errors. MLflow dashboard

opens at http://localhost:5000. Frontend opens at http://localhost:3000. Backend opens at http://localhost:8000.

## Aadi and Sanjan — ML Layer (Start: Week 1, finish by end of Week 3)

You two are responsible for everything that happens to a model — tracking it, judging it, comparing versions, and explaining changes in plain English.

## What you need to understand first

When the backend receives a user's training script, it runs it inside Docker. After the script finishes, a file called model.pkl appears in a folder. Your job starts here — you take that model.pkl, log its metrics, decide if it's good enough, and compare it against previous versions.

## Aadi — MLflow Tracking + Quality Gate

File:

ml/tracker.py

This file connects to MLflow and logs every training run. Think of MLflow as a diary that records every experiment automatically.

What this file must do:

- Start an MLflow run with a given run name


- Log parameters (what settings the user chose)

- Log metrics (accuracy, F1, precision, recall — whatever the training script produces)

- Log the model artifact ( model.pkl) so MLflow stores it

- End the run and return the run ID

```
\# tracker.py does things like:
import mlflow
deflog_run(run_name, params, metrics, model_path):
with mlflow.start_run(run_name=run_name)as run:
mlflow.log_params(params)
mlflow.log_metrics(metrics)
mlflow.log_artifact(model_path)
return run.info.run_id
```

File: ml/quality_gate.py

This is the gatekeeper. It takes the metrics from the training run and the criteria the user set, and returns either PASS or FAIL with a reason.

What this file must do:

- Accept metrics dict (e.g. {"accuracy": 0.87, "f1": 0.85})

- Accept criteria dict (e.g. {"accuracy": 0.85, "f1": 0.80})

- Compare every metric against every criterion

- Return {"passed": True} or {"passed": False, "reason": "accuracy 0.72 did not meet threshold 0.85"}

```
\# quality_gate.py does things like:
defcheck(metrics, criteria):
for metric, threshold in criteria.items():
if metrics.get(metric,0)< threshold:
return{"passed":False,"reason":f"{metric}{metrics[metric]}
did not meet threshold {threshold}"}
return{"passed":True}
```

File: ml/sample_models/iris_train.py

This is a fake ML engineer's script that you write just to test your own platform. It trains a simple model on the Iris dataset and saves model.pkl. You will also write titanic_train.py and

nslkdd_train.py the same way.

Every sample script must follow this rule at the end:


```
import pickle
withopen("model.pkl","wb")as f:
pickle.dump(model, f)
# Also print metrics so backend can read them
print(f"METRICS:accuracy={accuracy},f1={f1_score}")
```

When Aadi is done: Ebin and Joel can call tracker.log_run(...) and quality_gate.check(...) from the backend without knowing how they work internally.

## Sanjan — Version Comparison + AI Diagnosis

You start your work after Aadi has tracker.py working (around Week 2).

File: ml/comparator.py

This file compares two versions of a model — the current live one and the new one just trained. It uses Evidently AI to compare datasets and plain Python to compare metrics.

What this file must do:

- Accept old metrics and new metrics

- Accept old dataset path and new dataset path

- Compare metrics (which went up, which went down, by how much)

- Use Evidently to compare data distributions (did the dataset change significantly?)

- Return a structured comparison report as a Python dict

```
\# comparator.py does things like:
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
defcompare(old_metrics, new_metrics, old_data, new_data):
metric_diff ={}
for key in new_metrics:
metric_diff[key]={
"old": old_metrics.get(key),
"new": new_metrics.get(key),
"change": new_metrics[key]- old_metrics.get(key,0)
}
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=old_data, current_data=new_data)
```


```
drift_result = report.as_dict()
return{
"metric_diff": metric_diff,
"data_drift_detected": drift_result["metrics"][0]["result"]
["dataset_drift"]
}
```

File: ml/diagnosis.py

This is the AI explanation feature — your most important and unique contribution to the project.

What this file must do:

- Take the comparison report from comparator.py

- Build a clear prompt describing what changed

- Call an LLM API (Claude or OpenAI) with that prompt

- Return the plain English explanation as a string

```
\# diagnosis.py does things like:
import anthropic
defexplain(comparison_report):
client = anthropic.Anthropic(api_key="YOUR_KEY")
prompt =f"""
An ML engineer uploaded a new version of their model. Here is what
changed:
Metric changes: {comparison_report['metric_diff']}
Data drift detected: {comparison_report['data_drift_detected']}
In 3-4 sentences, explain to the ML engineer why their model likely
got better or worse, and what they should do next.
"""
message = client.messages.create(
model="claude-sonnet-4-6",
max_tokens=300,
messages=[{"role":"user","content": prompt}]
)
return message.content[0].text
```


When Sanjan is done: The backend can call comparator.compare(...) and diagnosis.explain(...) and show the result to the user on the dashboard.

## Ebin and Joel — Backend + Infrastructure (Start: Week 1, finish by end of Week 4)

You two are responsible for everything the user cannot see — receiving files, running training jobs, managing Docker containers, serving models, and returning links.

Split the work like this: Joel owns the API routes and file handling. Ebin owns Docker, deployment, and Nginx.

## Joel — API Routes and Pipeline Orchestration

File: backend/main.py

This is the entry point for the entire backend. It creates the FastAPI app and includes all the routes.

```
from fastapi import FastAPI
from routes import upload, models, pipeline
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
allow_headers=["*"])
app.include_router(upload.router)
app.include_router(models.router)
app.include_router(pipeline.router)
```

File:

backend/routes/upload.py

This handles the file upload from the frontend. The ML engineer sends their .py script, dataset .csv, and criteria through this endpoint.

What this route must do:

- Accept a POST /upload request with three things: training script, dataset file, criteria JSON

- Save the files to a temporary folder with a unique job ID


- Trigger the pipeline (call runner.py) as a background task

- Immediately return {"job_id": "abc123", "status": "running"} so the UI doesn't freeze

```
\# upload.py does things like:
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
import uuid, os
router = APIRouter()
@router.post("/upload")
asyncdefupload(background_tasks: BackgroundTasks,
script: UploadFile = File(...),
dataset: UploadFile = File(...),
criteria:str= Form(...)):
job_id =str(uuid.uuid4())
job_dir =f"jobs/{job_id}"
os.makedirs(job_dir)
# save files to job_dir
# background_tasks.add_task(run_pipeline, job_id, job_dir, criteria)
return{"job_id": job_id,"status":"running"}
```

File: backend/routes/models.py

These are the read endpoints — the frontend calls these to show the user their results.

## Endpoints to build:

- GET /models — return list of all models with their status and metrics

- GET /models/{model_id} — return one model's full details, link, metrics, comparison report, AI diagnosis

- GET /jobs/{job_id}/status — return whether a job is still running, passed, or failed

```
File: backend/services/runner.py
```

This is the most important backend file. It takes the user's training script and runs it inside a Docker container safely.

What this file must do:

- Take the job directory path (which has train.py and dataset.csv inside)

- Start a Docker container that mounts that folder

- Run python train.py inside the container

- Wait for it to finish


- Read the output to extract metrics

- Return the metrics and the path to model.pkl

```
\# runner.py does things like:
import subprocess
defrun_training(job_dir):
result = subprocess.run([
"docker","run","--rm",
"-v",f"{job_dir}:/workspace",
"-w","/workspace",
"automlops-runner", # this is the Docker image Ebin builds
"python","train.py"
], capture_output=True, text=True, timeout=300)
# parse metrics from stdout
# find model.pkl in job_dir
return metrics, model_path
```

File: backend/services/evaluator.py

This calls Aadi's quality_gate.py and decides what to do next. It's the bridge between the backend and the ML layer.

```
from ml.quality_gate import check
defevaluate(metrics, criteria):
result = check(metrics, criteria)
return result # {"passed": True} or {"passed": False, "reason": "..."}
```

## Ebin — Docker + Deployment + Nginx

File: backend/docker/Dockerfile.runner

This is the container that safely runs the ML engineer's training script. It must have Python and common ML libraries pre-installed.

```
FROM python:3.10-slim
RUN pip install pandas scikit-learn numpy xgboost lightgbm
WORKDIR /workspace
```


Build this image once with: docker build -t automlops-runner -f docker/Dockerfile.runner .

File: backend/docker/Dockerfile.serving

This is the container that serves a trained model as an API. It runs a tiny FastAPI app that loads model.pkl and exposes a /predict endpoint.

```
FROM python:3.10-slim
RUN pip install fastapi uvicorn scikit-learn pandas pickle5
COPY serve.py /app/serve.py
COPY model.pkl /app/model.pkl
WORKDIR /app
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

File: backend/services/deployer.py

This deploys a model by spinning up a Docker container using Dockerfile.serving. Every model gets its own container on a different port.

What this file must do:

- Accept a model_id and path to model.pkl

- Copy the model into a serving directory

- Start a Docker container for it on a free port

- Register the model ID → port mapping

- Return the serving port

```
\# deployer.py does things like:
import subprocess, json
PORT_REGISTRY ={} # model_id -> port number
defdeploy(model_id, model_path):
port = find_free_port()
subprocess.Popen([
"docker","run","-d",
"--name",f"model_{model_id}",
"-p",f"{port}:8080",
"-v",f"{model_path}:/app/model.pkl",
"automlops-serving"
])
PORT_REGISTRY[model_id]= port
return port
```


Nginx sits in front of all the serving containers and routes requests. This is what makes the link stable — when you update a model, you only change the Nginx config to point to the new container, but the URL stays the same.

```
server{
listen80;
location /models/abc123/predict{
proxy_pass http://localhost:8081; # points to the current live
container
}
location /models/xyz456/predict{
proxy_pass http://localhost:8082;
}
}
```

File: docker-compose.yml

This starts all the services together with one command ( docker-compose up).

```
version:"3"
services:
backend:
build: ./backend
ports:
-"8000:8000"
volumes:
- ./jobs:/app/jobs
- /var/run/docker.sock:/var/run/docker.sock # lets backend control
Docker
mlflow:
image: ghcr.io/mlflow/mlflow
ports:
-"5000:5000"
command: mlflow server --host 0.0.0.0
nginx:
image: nginx
ports:
-"80:80"
```


volumes:

- \- ./nginx/nginx.conf:/etc/nginx/nginx.conf

When Ebin and Joel are done: Aadi and Sanjan can test their ML scripts through the full pipeline by uploading them through the UI.

## Kevin — Frontend (Start: Week 2, finish by end of Week 5)

You build the web interface that the ML engineer actually sees and uses. You start in Week 2 because by then Joel will have the backend routes ready for you to connect to.

Tech stack: React + Tailwind CSS

File: frontend/src/pages/Upload.jsx

This is the main page — the first thing an ML engineer sees.

What it must have:

- File picker for training script ( .py files only)

- File picker for dataset ( .csv files only)

- Form fields for success criteria: accuracy threshold, F1 threshold

- A "Submit" button

- After submit → show a loading spinner with status updates ("Training in progress...", "Evaluating model...")

- When done → show result: PASSED with link, or FAILED with reason

```
// Upload.jsx rough structure
functionUploadPage(){
const[script, setScript]=useState(null)
const[dataset, setDataset]=useState(null)
const[criteria, setCriteria]=useState({accuracy:0.85,f1:0.80})
const[status, setStatus]=useState(null)
asyncfunctionhandleSubmit(){
const formData =newFormData()
formData.append("script", script)
formData.append("dataset", dataset)
formData.append("criteria",JSON.stringify(criteria))
const res =awaitfetch("http://localhost:8000/upload",{method:"POST",
```


```
body: formData })
const{ job_id }=await res.json()
pollStatus(job_id) // keep checking until done
}
// ...
}
```

File: frontend/src/pages/Dashboard.jsx

This page shows all past runs in a table — each row is a model version with its metrics and status (passed/failed).

What it must show:

- Table of all runs: version number, date, accuracy, F1, status (green/red badge)

- Clicking a row goes to ModelDetail page

```
File: frontend/src/pages/ModelDetail.jsx
```

This page shows everything about one model version — its metrics, its deployed link, and if it's a new version, the comparison against the previous one plus the AI diagnosis.

What it must show:

- Model metrics (accuracy, F1 etc.)

- Deployed link (copyable, with a test button)

- Comparison table (old vs new metrics, highlighted)

- AI Diagnosis box (the plain English explanation from Sanjan's module)

File: frontend/src/components/ComparisonTable.jsx

A reusable table component that shows old metrics vs new metrics side by side with color coding (green if improved, red if worse).

File: frontend/src/components/DiagnosisReport.jsx

A styled box that shows the AI explanation text. Keep it clean — white card, slightly different background, maybe a small AI icon.

When Kevin is done: The entire project is usable end-to-end through the browser.

## Phase Timeline (5 Weeks)


```
Week 1:
All → Set up local environment, install all tools, run MLflow dashboard
Aadi → Start tracker.py and quality_gate.py
Joel → Start main.py, upload route skeleton
Ebin → Build Dockerfile.runner, test it manually
Sanjan → Study Evidently AI docs, understand comparator.py structure
Kevin → Set up React project, build Upload.jsx layout (no real API yet)
Week 2:
Aadi → Finish tracker.py and quality_gate.py, write iris_train.py for
testing
Joel → Finish upload.py route and runner.py (can now receive files and
run them)
Ebin → Build Dockerfile.serving, build deployer.py
Sanjan → Start comparator.py
Kevin → Connect Upload.jsx to real backend upload endpoint
Week 3:
Aadi → Write titanic_train.py and nslkdd_train.py, test full pipeline end
to end
Joel → Finish models.py routes (GET endpoints for frontend to read)
Ebin → Finish Nginx config, test stable link with two model versions
Sanjan → Finish comparator.py, start diagnosis.py
Kevin → Build Dashboard.jsx
Week 4:
Aadi → Help test the full pipeline, fix edge cases in quality_gate.py
Joel → Integrate comparator and diagnosis into pipeline flow
Ebin → Finish docker-compose.yml, test full system startup with one
command
Sanjan → Finish diagnosis.py, test AI explanations with real comparison
data
Kevin → Build ModelDetail.jsx with comparison table and diagnosis display
Week 5:
All → End to end testing with all three datasets
All → Fix bugs
All → Prepare demo: upload → train → deploy → get link → reupload → see
comparison + diagnosis
Kevin → Polish UI, make it look clean
```

## Dependency Order (who waits for who)


```
Aadi finishes tracker.py + quality_gate.py
↓
Joel can integrate evaluator.py into the pipeline
↓
Ebin can test deployer.py with a real passing model
↓
Kevin can show real results in ModelDetail.jsx
Sanjan finishes comparator.py
↓
Sanjan finishes diagnosis.py
↓
Joel integrates both into the pipeline
↓
Kevin shows comparison + diagnosis on the frontend
```

## What the final demo looks like

- ý. Open the web UI in browser

- 2. Upload iris_train.py + iris.csv + set accuracy threshold to 85%

- 3. Watch the dashboard — status goes from "Running" to "Passed"

- 4. Copy the deployed link, paste it in Postman, send a prediction request, get result back

- 5. Upload iris_train_v2.py with slightly different parameters

- 6. Dashboard shows comparison table — which metrics went up and which went down

- 7. AI Diagnosis box shows: "Your new model improved accuracy by 2%. The feature scaling change you made helped the model generalise better."

- 8. The original link still works — now serving the better model underneath

## Questions to ask each other before coding

- Aadi to Joel: "What format do you want metrics returned in from the training script?"

- Joel to Ebin: "What folder structure should I give you so deployer.py can find model.pkl?"

- Ebin to Kevin: "What exact URL format should I use for deployed links so you can display them?"

- Sanjan to Joel: "When should I run the comparison — before or after the quality gate?"

- Kevin to Joel: "What does the JSON look like that the GET /models endpoint returns?"


Agreeing on these things early prevents everyone from having to rewrite code later.
