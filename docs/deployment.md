# Deployment Guide

This guide covers deploying Hermes to various environments.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Providers](#cloud-providers)
- [Configuration](#configuration)
- [Monitoring](#monitoring)

## Local Development

### Prerequisites

- Python 3.11+
- Poetry
- Docker & Docker Compose

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/hermes.git
cd hermes

# Install dependencies
poetry install --extras all

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Start dependencies
docker-compose up -d redis chromadb

# Run application
poetry run uvicorn hermes.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t hermes:latest .

# Run container
docker run -d \
  --name hermes \
  -p 8000:8000 \
  --env-file .env \
  hermes:latest
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Scale TTS workers
docker-compose up -d --scale tts-worker=3
```

### Docker Compose with Monitoring

```bash
# Start with Prometheus and Grafana
docker-compose --profile monitoring up -d
```

Access:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3 (optional)

### Using kubectl

#### 1. Create Namespace

```bash
kubectl create namespace hermes
```

#### 2. Create ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hermes-config
  namespace: hermes
data:
  APP_ENV: "production"
  LOG_LEVEL: "INFO"
  DEEPGRAM_MODEL: "nova-2"
  GEMINI_MODEL: "gemini-1.5-flash"
  CHROMADB_HOST: "chromadb"
  CHROMADB_PORT: "8002"
  # ... other non-sensitive config
```

#### 3. Create Secrets

```bash
# Create secrets from env file
kubectl create secret generic hermes-secrets \
  --from-env-file=.env \
  --namespace=hermes
```

Or manually:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hermes-secrets
  namespace: hermes
type: Opaque
data:
  DEEPGRAM_API_KEY: <base64-encoded-key>
  GEMINI_API_KEY: <base64-encoded-key>
  TWILIO_ACCOUNT_SID: <base64-encoded-sid>
  # ... other secrets
```

#### 4. Deploy Application

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hermes-app
  namespace: hermes
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hermes
  template:
    metadata:
      labels:
        app: hermes
    spec:
      containers:
      - name: app
        image: hermes:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: hermes-config
        - secretRef:
            name: hermes-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: hermes-service
  namespace: hermes
spec:
  selector:
    app: hermes
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hermes-ingress
  namespace: hermes
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  rules:
  - host: hermes.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hermes-service
            port:
              number: 80
```

Apply:

```bash
kubectl apply -f k8s/
```

#### 5. Deploy Dependencies

Redis:

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis --namespace hermes
```

PostgreSQL:

```bash
helm install postgres bitnami/postgresql --namespace hermes
```

ChromaDB:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chromadb
  namespace: hermes
spec:
  replicas: 1
  selector:
    matchLabels:
      app: chromadb
  template:
    metadata:
      labels:
        app: chromadb
    spec:
      containers:
      - name: chromadb
        image: chromadb/chroma:latest
        ports:
        - containerPort: 8000
        env:
        - name: IS_PERSISTENT
          value: "TRUE"
        volumeMounts:
        - name: chromadb-data
          mountPath: /chroma/chroma
      volumes:
      - name: chromadb-data
        persistentVolumeClaim:
          claimName: chromadb-pvc
```

### Using Helm

Create `Chart.yaml`:

```yaml
apiVersion: v2
name: hermes
description: Hermes Voice Support Service
type: application
version: 0.1.0
appVersion: "0.1.0"
```

Install:

```bash
helm install hermes ./helm-chart --namespace hermes
```

## Cloud Providers

### AWS

#### ECS (Elastic Container Service)

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name hermes

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster hermes \
  --service-name hermes-app \
  --task-definition hermes:1 \
  --desired-count 2 \
  --launch-type FARGATE
```

#### EKS (Elastic Kubernetes Service)

```bash
# Create EKS cluster
eksctl create cluster \
  --name hermes \
  --region us-west-2 \
  --node-type t3.medium \
  --nodes 3

# Deploy
kubectl apply -f k8s/
```

### Google Cloud Platform

#### GKE (Google Kubernetes Engine)

```bash
# Create cluster
gcloud container clusters create hermes \
  --zone us-central1-a \
  --num-nodes 3

# Get credentials
gcloud container clusters get-credentials hermes --zone us-central1-a

# Deploy
kubectl apply -f k8s/
```

#### Cloud Run (Serverless)

```bash
# Deploy to Cloud Run
gcloud run deploy hermes \
  --image gcr.io/PROJECT_ID/hermes:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "APP_ENV=production"
```

Note: Cloud Run has WebSocket limitations. Use GKE for production WebSocket support.

### Azure

#### AKS (Azure Kubernetes Service)

```bash
# Create resource group
az group create --name hermes-rg --location eastus

# Create cluster
az aks create \
  --resource-group hermes-rg \
  --name hermes \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group hermes-rg --name hermes

# Deploy
kubectl apply -f k8s/
```

## Configuration

### Environment Variables

See `.env.example` for all available options.

Critical variables:
- `APP_ENV`: Environment (development, staging, production)
- `DEEPGRAM_API_KEY`: Deepgram API key
- `GEMINI_API_KEY`: Gemini API key
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

### Secrets Management

#### Kubernetes Secrets

```bash
kubectl create secret generic hermes-secrets \
  --from-literal=DEEPGRAM_API_KEY=xxx \
  --from-literal=GEMINI_API_KEY=xxx \
  --namespace=hermes
```

#### AWS Secrets Manager

```bash
aws secretsmanager create-secret \
  --name hermes/production \
  --secret-string file://secrets.json
```

#### HashiCorp Vault

```bash
vault kv put secret/hermes \
  DEEPGRAM_API_KEY=xxx \
  GEMINI_API_KEY=xxx
```

## Monitoring

### Prometheus

Prometheus is automatically configured to scrape metrics from `/metrics` endpoint.

Key metrics:
- `hermes_active_calls`: Number of active calls
- `hermes_call_duration_seconds`: Call duration histogram
- `hermes_stt_latency_seconds`: STT latency
- `hermes_llm_latency_seconds`: LLM latency
- `hermes_tts_latency_seconds`: TTS latency

### Grafana Dashboard

Import dashboard from `monitoring/grafana/dashboards/hermes.json`.

Dashboards include:
- Overview: Active calls, error rates, latency
- Performance: Service latencies, throughput
- Errors: Error rates by service and type

### Alerts

Example Prometheus alerts:

```yaml
groups:
  - name: hermes-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(hermes_calls_total{status="failed"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in Hermes"

      - alert: HighLatency
        expr: hermes_llm_latency_seconds{quantile="0.95"} > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM latency"

      - alert: ServiceDown
        expr: up{job="hermes"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Hermes service is down"
```

### Logging

Logs are structured JSON in production:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "hermes.websocket.handler",
  "message": "call_connected",
  "call_sid": "CA1234567890",
  "stream_sid": "MZ1234567890"
}
```

Send logs to:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Datadog
- Splunk
- CloudWatch (AWS)
- Stackdriver (GCP)

## Troubleshooting

### Common Issues

#### WebSocket Connection Fails

```bash
# Check if ingress supports WebSocket
kubectl describe ingress hermes-ingress

# Verify service endpoints
kubectl get endpoints hermes-service
```

#### High Latency

1. Check service health:
   ```bash
   curl http://hermes/ready
   ```

2. Review metrics:
   ```bash
   curl http://hermes/metrics | grep latency
   ```

3. Check resource usage:
   ```bash
   kubectl top pods -n hermes
   ```

#### Database Connection Issues

```bash
# Test connection
kubectl exec -it hermes-pod -- python -c "
from hermes.models import init_db
init_db()
"

# Check logs
kubectl logs hermes-pod | grep -i database
```

## Security Checklist

- [ ] API keys stored in secrets (not ConfigMap)
- [ ] TLS enabled for all external traffic
- [ ] Network policies configured
- [ ] Pod security policies applied
- [ ] Resource limits set
- [ ] Health checks configured
- [ ] Logging enabled
- [ ] Monitoring configured
- [ ] Backup strategy for databases
- [ ] Disaster recovery plan documented
