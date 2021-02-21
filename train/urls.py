from django.urls import path
from .views import IrisTrain
from django.conf.urls import url

urlpatterns = [
    url(r'^iris/$', IrisTrain.as_view(), name="IrisTrain"),
]