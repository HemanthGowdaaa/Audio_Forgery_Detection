from django.urls import path
from detector import views

urlpatterns = [
    # Legacy routes (React frontend compatibility)
    path("health", views.HealthCheckView.as_view(), name="legacy-health"),
    path("health/", views.HealthCheckView.as_view(), name="legacy-health-slash"),
    path("status", views.ModelStatusView.as_view(), name="legacy-status"),
    path("status/", views.ModelStatusView.as_view(), name="legacy-status-slash"),
    path("predict", views.PredictView.as_view(), name="legacy-predict"),
    path("predict/", views.PredictView.as_view(), name="legacy-predict-slash"),
    path("upload", views.UploadView.as_view(), name="legacy-upload"),
    path("upload/", views.UploadView.as_view(), name="legacy-upload-slash"),
    path("metrics", views.MetricsView.as_view(), name="legacy-metrics"),
    path("metrics/", views.MetricsView.as_view(), name="legacy-metrics-slash"),
    path("report", views.ReportView.as_view(), name="legacy-report"),
    path("report/", views.ReportView.as_view(), name="legacy-report-slash"),

    # Versioned v1 routes
    path("api/v1/health/", views.HealthCheckView.as_view(), name="v1-health"),
    path("api/v1/health", views.HealthCheckView.as_view(), name="v1-health-no-slash"),
    path("api/v1/model-status/", views.ModelStatusView.as_view(), name="v1-model-status"),
    path("api/v1/model-status", views.ModelStatusView.as_view(), name="v1-model-status-no-slash"),
    path("api/v1/status/", views.ModelStatusView.as_view(), name="v1-status"),
    path("api/v1/status", views.ModelStatusView.as_view(), name="v1-status-no-slash"),
    path("api/v1/predict/", views.PredictView.as_view(), name="v1-predict"),
    path("api/v1/predict", views.PredictView.as_view(), name="v1-predict-no-slash"),
    path("api/v1/upload/", views.UploadView.as_view(), name="v1-upload"),
    path("api/v1/upload", views.UploadView.as_view(), name="v1-upload-no-slash"),
    path("api/v1/metrics/", views.MetricsView.as_view(), name="v1-metrics"),
    path("api/v1/metrics", views.MetricsView.as_view(), name="v1-metrics-no-slash"),
    path("api/v1/report/", views.ReportView.as_view(), name="v1-report"),
    path("api/v1/report", views.ReportView.as_view(), name="v1-report-no-slash"),
]
