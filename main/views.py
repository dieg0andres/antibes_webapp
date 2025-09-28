

from django.shortcuts import render
from django.http import JsonResponse

from main.views_helpers.todoist_helpers import build_dashboard


# Create your views here.
def index(request):
    return render(request, 'main/index.html')


def todoist(request):

    dashboard_data = build_dashboard()
    return render(request, 'main/todoist.html', {'dashboard_data': dashboard_data})