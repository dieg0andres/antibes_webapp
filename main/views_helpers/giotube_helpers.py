from datetime import datetime
from pathlib import Path

from django.conf import settings


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

