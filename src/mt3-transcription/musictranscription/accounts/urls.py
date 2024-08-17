from django.urls import path
from .views import UserCreate
from . import views

urlpatterns = [
    path('signup/', UserCreate.as_view(), name='signup'),
    path('login/', views.LoginView.as_view(), name='login'),
]
