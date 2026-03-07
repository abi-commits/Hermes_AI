# Deployment Guide (Production)

This guide covers deploying the Hermes AI application to production using [Modal](https://modal.com) as the infrastructure provider. Modal provides serverless infrastructure for Python, making it easy to deploy AI and data applications without managing servers.

## Prerequisites

1.  **Modal Account:** Sign up for an account at [modal.com](https://modal.com).
2.  **Modal CLI:** Ensure the Modal client is installed in your environment.
    ```bash
    pip install modal
    ```
    *(Or if using `uv` as configured in the project: `uv pip install modal`)*
3.  **Authentication:** Authenticate the CLI with your Modal account.
    ```bash
    modal setup
    ```

## 1. Secrets Management

Hermes AI relies on several API keys (e.g., Twilio, LLM providers, Speech-to-Text, Text-to-Speech). In Modal, these are managed using **Modal Secrets**.

You need to create a Secret in the Modal dashboard (or via CLI) containing your production environment variables.

### Creating Secrets via Modal Dashboard

1. Navigate to the **Secrets** page in your Modal workspace.
2. Click **Create new secret**.
3. Select **Custom** or import from a `.env` file.
4. Name the secret `hermes-secrets` (or similar).
5. Add the necessary key-value pairs from your `.env` file (e.g., `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `OPENAI_API_KEY`, etc.).

### Using Secrets in Code

The Modal app configuration (typically in a file like `modal_app.py` or `hermes/main.py` if adapted for Modal) will mount this secret:

```python
import modal

app = modal.App("hermes-ai")

@app.function(secrets=[modal.Secret.from_name("hermes-secrets")])
def ...
```

## 2. Environment and Dependencies

Modal builds a container image based on your project's dependencies. Since this project uses `uv` and `pyproject.toml`, the Modal image definition should reflect this to ensure the correct environment is built.

Example Modal Image definition:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install -r pyproject.toml --system")
    # Add any other system dependencies (e.g., ffmpeg if required for audio)
    # .apt_install("ffmpeg")
)
```

## 3. Deployment Workflow

### Local Development / Testing (Ephemeral)

To test your deployment iteratively without creating a permanent production version, use the `serve` command. This will watch your local files and live-reload the application on Modal's infrastructure.

```bash
modal serve modal_app.py
```
*Note: Replace `modal_app.py` with the actual entry point script for your Modal application.*

### Production Deployment

Once you are satisfied with the local testing, deploy the application to production. This creates a permanent version of your app and provides production URLs.

```bash
modal deploy modal_app.py
```

## 4. Webhook Configuration (Twilio)

After running `modal deploy`, Modal will output a production URL for your web endpoints (e.g., `https://your-workspace--hermes-ai-web.modal.run`).

You must update your Twilio phone number configuration to point to this new production URL:

1. Log in to the Twilio Console.
2. Go to **Phone Numbers** -> **Manage** -> **Active Numbers**.
3. Select the phone number used for Hermes AI.
4. Scroll down to the **Voice & Fax** section.
5. In the **"A CALL COMES IN"** configuration, set the webhook URL to the Modal production URL (e.g., `https://your-workspace--hermes-ai-web.modal.run/api/twilio/incoming`).
6. Save the changes.

## 5. Monitoring & Logs

- **Modal Dashboard:** Real-time logs and performance metrics (CPU, Memory, execution time) for your endpoints are available in the Modal Dashboard under your App.
- **Application Metrics:** (Optional) If you have configured Prometheus/Grafana (as seen in the `monitoring/` directory), ensure the metrics endpoint is exposed and accessible by your Prometheus server or configure a remote write from within the Modal app.