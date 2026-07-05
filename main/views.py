import json
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from main.views_helpers.todoist_helpers import build_dashboard, pull_old_tasks_to_today
from main.views_helpers.pay_mortgage_helpers import build_mortgage_dashboard
from main.views_helpers.giotube_helpers import build_giotube_playlist
from main.views_helpers.personal_finance_dashboard_helpers import (
    build_personal_finance_dashboard_context,
)
from main.views_helpers.trading_helpers import build_trading_dashboard_context
from main.views_helpers.volatility_dashboard_helpers import (
    get_volatility_dashboard_payload,
    get_volatility_dashboard_graphs_payload,
)
from main.views_helpers.volatility_dashboard_helpers import get_volatility_dashboard_graphs_ui_payload
from main.views_helpers.tag2_0_leaderboard_helpers import (
    LeaderboardRequestError,
    get_leaderboard,
    submit_leaderboard_score,
)



# Create your views here.
def index(request):
    return render(request, 'main/index.html')


@require_http_methods(["GET", "POST"])
def todoist(request):
    if request.method == 'POST':
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
            return JsonResponse(
                {'status': 'error', 'message': 'Missing action'},
                status=400
            )

        if action == 'pull_old_tasks_to_today':
            updated_tasks = pull_old_tasks_to_today()
            dashboard_data = build_dashboard()

            return JsonResponse({
                'status': 'success',
                'message': f'Updated {len(updated_tasks)} old tasks to today',
                'dashboard_data': dashboard_data,
            })

        return JsonResponse(
            {'status': 'error', 'message': f'Unknown action: {action}'},
            status=400
        )

    dashboard_data = build_dashboard()
    return render(request, 'main/todoist.html', {'dashboard_data': dashboard_data})

def pay_mortgage(request):
    dashboard_data = build_mortgage_dashboard()
    return render(request, 'main/pay_mortgage.html', {'dashboard_data': dashboard_data})


def giotube(request):
    return render(request, 'main/giotube.html', {'videos': build_giotube_playlist()})


def tag2_0(request):
    media_base_url = f"{settings.MEDIA_URL}tag2_0/"
    characters = [
        {
            "name": "Ducky",
            "image": f"{media_base_url}pictures/ducky.png",
            "video": f"{media_base_url}videos/ducky_intro_video.mp4",
            "description": "A fast, unpredictable hunter built for tight turns and sudden pressure.",
        },
        {
            "name": "Malice",
            "image": f"{media_base_url}pictures/malice.png",
            "video": f"{media_base_url}videos/malice_intro_video.mp4",
            "description": "An animal-like killer... capable of morphing into a tiger or dino all while chasing you.",
        },
        {
            "name": "Show Runner",
            "image": f"{media_base_url}pictures/show_runner.png",
            "video": f"{media_base_url}videos/show_runner_intro_video.mp4",
            "description": "Keeps the arena moving, the tension climbing, and the spotlight on you.",
        },
        {
            "name": "Sub Slasher",
            "image": f"{media_base_url}pictures/sub_slasher.png",
            "video": f"{media_base_url}videos/sub_slasher_intro_video.mp4",
            "description": "Cuts through you with precision with his popscicle knife... has no patience for mistakes.",
        },
        {
            "name": "Vengeance Bot",
            "image": f"{media_base_url}pictures/vengance_bot.png",
            "video": f"{media_base_url}videos/vengance_bot_intro_video.mp4",
            "description": "A relentless machine that learns the rhythm of your panic.",
        },
    ]
    context = {
        "download_url": f"{media_base_url}downloads/Tag%202.0-mac.zip",
        "hero_video_url": f"{media_base_url}videos/tag2_0_promo.mp4",
        "gameplay_video_url": f"{media_base_url}videos/game_intro_video.mp4",
        "characters": characters,
        "survivor": {
            "name": "Survivor",
            "image": f"{media_base_url}pictures/survivor.png",
            "description": "Your only job: move first, stay sharp, and survive one more chase.",
        },
        "preview_videos": [
            {
                "title": "Arcade Chase",
                "url": f"{media_base_url}videos/tag2_0_promo.mp4",
                "caption": "A quick hit of the speed, pressure, and retro horror tone.",
            },
            {
                "title": "Game Intro",
                "url": f"{media_base_url}videos/game_intro_video.mp4",
                "caption": "Step into the arena and get a feel for the rules fast.",
            },
            {
                "title": "Show Runner",
                "url": f"{media_base_url}videos/show_runner_intro_video.mp4",
                "caption": "The chase gets louder when the show starts.",
            },
        ],
    }
    return render(request, "main/tag2_0.html", context)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def tag2_0_leaderboard(request):
    if request.method == "GET":
        return JsonResponse(get_leaderboard(), json_dumps_params={"indent": 2})

    try:
        response = submit_leaderboard_score(request)
    except LeaderboardRequestError as error:
        return JsonResponse(
            {"status": "error", "message": error.message},
            status=error.status_code,
        )

    return JsonResponse(response, json_dumps_params={"indent": 2})


def tag2_0_leaderboard_ui(request):
    return render(request, "main/tag2_0_leaderboard.html")


def trading_dashboard(request):
    context = build_trading_dashboard_context()
    return render(request, 'main/trading_dashboard.html', context)


@require_http_methods(["GET", "POST"])
def personal_finance_dashboard(request):
    force_refresh = request.GET.get("refresh") == "1"

    if request.method == "POST":
        action = None

        if request.content_type and "application/json" in request.content_type:
            try:
                payload = json.loads(request.body or b"{}")
            except json.JSONDecodeError:
                payload = {}
            action = payload.get("action")

        if not action:
            action = request.POST.get("action")

        if action == "refresh":
            force_refresh = True
        elif action:
            return JsonResponse(
                {"status": "error", "message": f"Unknown action: {action}"},
                status=400,
            )

    context = build_personal_finance_dashboard_context(
        force_refresh=force_refresh,
    )
    return JsonResponse(context, json_dumps_params={"indent": 2})


def personal_finance_dashboard_ui(request):
    return render(request, "main/personal_finance_dashboard.html")



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
    


def volatility_dashboard_graphs_ui(request):
    symbol = request.GET.get("symbol")
    rv_reference = request.GET.get("rv_reference", "primary20")
    target_dte_days = float(request.GET.get("target_dte_days", "30"))
    run_id = request.GET.get("run_id")
    run_id = int(run_id) if run_id else None
    range_str = request.GET.get("range", "1y")

    payload = get_volatility_dashboard_graphs_ui_payload(
        symbol=symbol,
        rv_reference=rv_reference,
        target_dte_days=target_dte_days,
        run_id=run_id,
        range=range_str,
    )
    payload["range_options"] = ["3m","6m","1y","3y","5y","max"]

    return render(request, "main/volatility_dashboard_graphs.html", payload)
