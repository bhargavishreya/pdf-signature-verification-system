from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'), 
    path('verify-pdf/no-change', views.verify_pdf_no_change, name='verify_pdf_no_change'),
    path('verify-pdf/change', views.verify_pdf_change, name='verify_pdf_change'),
    path('success/', views.success_page, name='success')
    

]
