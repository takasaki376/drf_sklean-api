from .views import IrisPredict
from django.conf.urls import url

urlpatterns = [
    # path('IrisPredict/', IrisPredict.as_view(), name='IrisPredict'),
    url('IrisPredict/', IrisPredict.as_view(), name='IrisPredict'),
]