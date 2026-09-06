# Deployment

## Local demonstration

The supported demo is Streamlit:

```powershell
python -m streamlit run app\streamlit_app.py
```

The current index manifest contains local Hugging Face cache paths, and the generated FAISS/embedding files are intentionally excluded from Git. The complete demo therefore runs on the machine where the dataset and index have been built.

## Vercel

Vercel is not the correct runtime for this application. Vercel expects a Python serverless entrypoint such as `api/index.py`; Streamlit is a long-running application server and cannot be converted into a working Streamlit GUI by adding a dummy Python entrypoint.

Deploying this repository directly to Vercel will produce:

```text
No python entrypoint found
```

Use Streamlit Community Cloud, Hugging Face Spaces, or another service that supports a persistent Streamlit process instead. A hosted deployment also requires a separate artifact/data strategy: build the FAISS index against deployable image paths or object storage rather than local Hugging Face cache paths.

## Recruiter handoff

For the current take-home, send the GitHub repository together with the local startup command and the audit/evaluation reports. Do not claim that a Vercel URL is the working model demo until the model artifacts and image storage have been made deployable.