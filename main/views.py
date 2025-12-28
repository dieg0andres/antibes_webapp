import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from main.views_helpers.todoist_helpers import build_dashboard, pull_old_tasks_to_today
from main.views_helpers.pay_mortgage_helpers import build_mortgage_dashboard
from main.views_helpers.giotube_helpers import build_giotube_playlist
from main.views_helpers.trading_helpers import build_trading_dashboard_context
from main.views_helpers.volatility_dashboard_helpers import (
    get_volatility_dashboard_payload,
    get_volatility_dashboard_graphs_payload,
)


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


def giotube(request):
    return render(request, 'main/giotube.html', {'videos': build_giotube_playlist()})


def trading_dashboard(request):
    context = build_trading_dashboard_context()
    return render(request, 'main/trading_dashboard.html', context)



def volatility_dashboard(request):
    symbol = request.GET.get("symbol")
    rv_reference = request.GET.get("rv_reference", "primary20")
    target_dte_days = float(request.GET.get("target_dte_days", "30"))
    run_id = request.GET.get("run_id")
    run_id = int(run_id) if run_id else None

    payload = get_volatility_dashboard_payload(
        symbol=symbol,
        rv_reference=rv_reference,
        target_dte_days=target_dte_days,
        run_id=run_id,
    )
    return JsonResponse(payload, json_dumps_params={"indent": 2})



def volatility_dashboard_ui(request):
    symbol = request.GET.get("symbol")
    rv_reference = request.GET.get("rv_reference", "primary20")
    target_dte_days = float(request.GET.get("target_dte_days", "30"))
    run_id = request.GET.get("run_id")
    run_id = int(run_id) if run_id else None

    payload = get_volatility_dashboard_payload(
        symbol=symbol,
        rv_reference=rv_reference,
        target_dte_days=target_dte_days,
        run_id=run_id,
    )

    # Render HTML; payload includes options + selected + snapshot + recent_session_bars
    return render(request, "main/volatility_dashboard.html", payload)



def volatility_dashboard_graphs(request):
    symbol = request.GET.get("symbol")
    rv_reference = request.GET.get("rv_reference", "primary20")
    target_dte_days = float(request.GET.get("target_dte_days", "30"))
    run_id = request.GET.get("run_id")
    run_id = int(run_id) if run_id else None

    payload = get_volatility_dashboard_graphs_payload(
        symbol=symbol,
        rv_reference=rv_reference,
        target_dte_days=target_dte_days,
        run_id=run_id,
    )
    return JsonResponse(payload, json_dumps_params={"indent": 2})