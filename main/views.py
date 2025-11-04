import json
import time
from datetime import datetime, date

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from main.views_helpers.todoist_helpers import build_dashboard, pull_old_tasks_to_today


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
    # Mortgage data table
    mortgage_data = {
        'Jul 2025': {'principal': 400000, 'interest_paid': 0},
        'Aug 2025': {'principal': 399641, 'interest_paid': 2250},
        'Sep 2025': {'principal': 399280, 'interest_paid': 4498},
        'Oct 2025': {'principal': 398028, 'interest_paid': 6744},
        'Nov 2025': {'principal': 390490, 'interest_paid': 8983},
        'Dec 2025': {'principal': 380908, 'interest_paid': 11179},
        'Jan 2026': {'principal': 380162, 'interest_paid': 13322},
        'Feb 2026': {'principal': 364412, 'interest_paid': 15460},
        'Mar 2026': {'principal': 348573, 'interest_paid': 17510},
        'Apr 2026': {'principal': 332644, 'interest_paid': 19471},
        'May 2026': {'principal': 316627, 'interest_paid': 21342},
        'Jun 2026': {'principal': 300519, 'interest_paid': 23123},
        'Jul 2026': {'principal': 284320, 'interest_paid': 24813},
        'Aug 2026': {'principal': 268031, 'interest_paid': 26413},
        'Sep 2026': {'principal': 251649, 'interest_paid': 27920},
        'Oct 2026': {'principal': 235176, 'interest_paid': 29336},
        'Nov 2026': {'principal': 218610, 'interest_paid': 30659},
        'Dec 2026': {'principal': 201951, 'interest_paid': 31889},
        'Jan 2027': {'principal': 185198, 'interest_paid': 33024},
        'Feb 2027': {'principal': 168351, 'interest_paid': 34066},
        'Mar 2027': {'principal': 151409, 'interest_paid': 35013},
        'Apr 2027': {'principal': 134372, 'interest_paid': 35865},
        'May 2027': {'principal': 117239, 'interest_paid': 36621},
        'Jun 2027': {'principal': 100009, 'interest_paid': 37280},
        'Jul 2027': {'principal': 82683, 'interest_paid': 37843},
        'Aug 2027': {'principal': 65259, 'interest_paid': 38308},
        'Sep 2027': {'principal': 47737, 'interest_paid': 38675},
        'Oct 2027': {'principal': 30117, 'interest_paid': 38943},
        'Nov 2027': {'principal': 12398, 'interest_paid': 39113},
        'Dec 2027': {'principal': 0, 'interest_paid': 39183},
    }
    
    # Get today's date
    today = date.today()
    
    # 1) Calculate days until Dec 1, 2027
    mortgage_end_date = date(2027, 12, 1)
    days_finish_mortgage = (mortgage_end_date - today).days
    
    # 2) Look up principal based on today's month and year
    current_month_year = today.strftime('%b %Y')
    print("Current month year: ", current_month_year)
    
    # Default values if current month not found (e.g., if we're past Dec 2027)
    principal = 0
    interest_paid = 39183  # Use the last known value
    
    if current_month_year in mortgage_data:
        principal = mortgage_data[current_month_year]['principal']
        interest_paid = mortgage_data[current_month_year]['interest_paid']
    
    dashboard_data = {
        'days_finish_mortgage': days_finish_mortgage,
        'principal': principal,
        'interest_paid': interest_paid,
    }
    
    return render(request, 'main/pay_mortgage.html', {'dashboard_data': dashboard_data})