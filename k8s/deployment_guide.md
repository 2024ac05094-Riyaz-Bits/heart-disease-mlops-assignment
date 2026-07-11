# Step 14 - Local Kubernetes Deployment Guide

This project uses a local Kubernetes deployment approach suitable for Docker Desktop Kubernetes or Minikube.

## Prerequisites

- Docker Desktop installed and running
- Kubernetes enabled in Docker Desktop, or Minikube installed
- `kubectl` available in terminal
- Docker image already built:

```powershell
docker build -t heart-disease-api .
```

## Option 1: Docker Desktop Kubernetes

1. Enable Kubernetes in Docker Desktop settings.
2. Confirm cluster access:

```powershell
kubectl get nodes
```

3. Apply manifests:

```powershell
kubectl apply -f k8s/heart-disease-deployment.yaml
kubectl apply -f k8s/heart-disease-service.yaml
```

4. Check deployment:

```powershell
kubectl get pods
kubectl get svc
```

5. Access API:

- URL: `http://localhost:30080/docs`

## Option 2: Minikube

1. Start Minikube:

```powershell
minikube start
```

2. Build the image inside Minikube Docker environment:

```powershell
minikube image load heart-disease-api:latest
```

3. Apply manifests:

```powershell
kubectl apply -f k8s/heart-disease-deployment.yaml
kubectl apply -f k8s/heart-disease-service.yaml
```

4. Get service URL:

```powershell
minikube service heart-disease-api-service --url
```

## Verification

Use either the browser `/docs` page or PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:30080/predict -ContentType "application/json" -Body (Get-Content artifacts/api/sample_request.json -Raw)
```

## Useful Commands

```powershell
kubectl get all
kubectl describe deployment heart-disease-api
kubectl logs deployment/heart-disease-api
kubectl delete -f k8s/heart-disease-service.yaml
kubectl delete -f k8s/heart-disease-deployment.yaml
```

## Notes for Report

- Deployment type: local Kubernetes
- Service type: NodePort
- Health endpoint used for probes: `/health`
- Suggested screenshot evidence:
  - `kubectl get pods`
  - `kubectl get svc`
  - browser opened at `/docs`
  - successful `/predict` response
