from django.urls import path
from .views import ask_document

urlpatterns = [
    path("ask/", ask_document, name="ask_document"),
]