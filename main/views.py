import json
from datetime import datetime
from pathlib import Path

from django.conf import settings
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


def giotube(request):
    return render(request, 'main/giotube.html', {'videos': build_giotube_playlist()})


def build_giotube_playlist():
    video_dir = Path(settings.MEDIA_ROOT) / 'videos'
    thumbnail_dir = Path(settings.MEDIA_ROOT) / 'thumbnails'
    video_extensions = {'.mp4', '.mov', '.m4v', '.webm', '.ogg'}
    thumbnail_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    placeholder = 'https://via.placeholder.com/360x202/111827/eeeeee?text=No+Thumbnail'

    if not video_dir.exists():
        return []

    videos = []
    for video_file in sorted(video_dir.iterdir(), key=lambda p: p.name.lower()):
        if not video_file.is_file() or video_file.suffix.lower() not in video_extensions:
            continue

        video_metadata = video_file.stat()
        uploaded_at = datetime.fromtimestamp(video_metadata.st_mtime).strftime('%b %d, %Y')
        size_mb = video_metadata.st_size / (1024 * 1024)

        thumbnail_url = placeholder
        if thumbnail_dir.exists():
            for ext in thumbnail_extensions:
                candidate = thumbnail_dir / f'{video_file.stem}{ext}'
                if candidate.exists():
                    thumbnail_url = f'{settings.MEDIA_URL}thumbnails/{candidate.name}'
                    break

        videos.append({
            'title': video_file.stem.replace('_', ' ').title(),
            'video_url': f'{settings.MEDIA_URL}videos/{video_file.name}',
            'thumbnail_url': thumbnail_url,
            'meta': f'{size_mb:.1f} MB • {uploaded_at}',
            'extension': video_file.suffix.lstrip('.').upper(),
            'badge': 'HD',
        })

    return videos