from django.urls import path

from . import views

urlpatterns = [
    path('score/', views.score, name='score'),
    path('train/', views.train, name='train'),
    path('validate/', views.validate, name='validate'),
    path('save/', views.save, name='save'),
    
    
    
]