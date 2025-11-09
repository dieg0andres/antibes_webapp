import json

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from main.views_helpers.todoist_helpers import build_dashboard, pull_old_tasks_to_today
from main.views_helpers.pay_mortgage_helpers import build_mortgage_dashboard


# Create your views here.
def index(request):
    return render(request, 'main/index.html')


@require_http_methods(["GET", "POST"])
def todoist(request):

    if request.method == 'POST':
        # Determine action from JSON body or form-encoded data
        action = None
        if request.content_type and 'application/json' in request.content_type:
            try:
                payload = json.loads(request.body or b"{}")
            except json.JSONDecodeError:
                payload = {}
            action = payload.get('action')
        if not action:
            action = request.POST.get('action')

        if not action:
            return JsonResponse({'status': 'error', 'message': 'Missing action'}, status=400)

        # Branch behavior based on action value
        if action == 'pull_old_tasks_to_today':
            pull_old_tasks_to_today()
    
    # Handle GET request (original functionality)
    dashboard_data = build_dashboard()
#    print("Dashboard data: ", dashboard_data)
    return render(request, 'main/todoist.html', {'dashboard_data': dashboard_data})


def pay_mortgage(request):
    dashboard_data = build_mortgage_dashboard()
    return render(request, 'main/pay_mortgage.html', {'dashboard_data': dashboard_data})