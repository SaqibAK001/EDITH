from django.urls import path
from .views import ask_document

from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

def home(request):
    return JsonResponse({
        "status": "running",
        "message": "EDITH Backend API"
    })

urlpatterns = [
    path("", home),
    path("api/", include("api.urls")),
    path("admin/", admin.site.urls),
]