"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from main import views
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    path('', views.index, name='index'),
    path('todoist/', views.todoist, name='todoist'),
    path('pay_mortgage/', views.pay_mortgage, name='pay_mortgage'),
    path('giotube/', views.giotube, name='giotube'),
    path('tag2_0/', views.tag2_0, name='tag2_0'),
    path('tag2_0/leaderboard/', views.tag2_0_leaderboard, name='tag2_0_leaderboard'),
    path('tag2_0/leaderboard/ui/', views.tag2_0_leaderboard_ui, name='tag2_0_leaderboard_ui'),
    path('personal_finance/dashboard/', views.personal_finance_dashboard, name='personal_finance_dashboard'),
    path('personal_finance/dashboard/ui/', views.personal_finance_dashboard_ui, name='personal_finance_dashboard_ui'),
    path('trading/dashboard/', views.trading_dashboard, name='trading_dashboard'),
    path('trading/volatility_dashboard/', views.volatility_dashboard, name='volatility_dashboard'),
    path('trading/volatility_dashboard/ui/', views.volatility_dashboard_ui, name='volatility_dashboard_ui'),
    path('trading/volatility_dashboard/graphs/', views.volatility_dashboard_graphs, name='volatility_dashboard_graphs'),
    path('trading/volatility_dashboard/graphs/ui/', views.volatility_dashboard_graphs_ui, name='volatility_dashboard_graphs_ui'),
    path('media/<path:path>', serve, {'document_root': settings.MEDIA_ROOT}),
    path('admin/', admin.site.urls),
]
