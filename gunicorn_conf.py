# gunicorn_conf.py - Gunicorn configuration (Python format)

############ Basic process / app settings ############
# The WSGI entry point: module:callable
wsgi_app = "config.wsgi:application"

# Working directory (so relative imports and manage.py are found)
chdir = "/home/diegogalindo/my_stuff/01_Projects/antibes_webapp"

# Bind to a unix socket (recommended) or to 127.0.0.1:8000
# Use unix socket for Nginx <-> Gunicorn communication
# bind = "unix:/run/gunicorn-antibes/gunicorn.sock"
# Alternative: bind = "127.0.0.1:8000"

# PID file (useful for debugging / scripting)
# pidfile = "/run/gunicorn-antibes/gunicorn.pid"

# (temporary, user-writable)
pidfile = "/tmp/gunicorn-antibes.pid"
# and optionally use a localhost bind
bind = "10.0.0.176:8000"


############ Process sizing ############
# Worker class: 'sync' is default for WSGI. 'gthread' enables threads.
worker_class = "sync"         # for Django WSGI apps this is fine; use "gthread" for threaded behavior
# Workers = (2 x CPU cores) + 1 is a common guideline
workers = 3
# Optional: threads per worker (if you use gthread, set worker_class='gthread' and threads>1)
threads = 2

# Graceful shutdown / timeouts
timeout = 30          # seconds to wait for workers to respond before restarting
graceful_timeout = 30
keepalive = 2         # seconds for keep-alive connections

# Max requests: recycle workers after handling this many requests to limit memory leaks
max_requests = 1000
max_requests_jitter = 50

############ Logging ############
# "-" means stdout/stderr (systemd/journal will capture)
accesslog = "-"       # or "/var/log/antibes/gunicorn.access.log"
errorlog = "-"        # or "/var/log/antibes/gunicorn.error.log"
loglevel = "info"

############ Performance / preload ############
# Preload app in master process before forking workers; reduces memory but can break some things.
preload_app = True

############ Security / permissions ############
# If you want Gunicorn to drop privileges after binding to socket, set user/group
# Note: when running under systemd we usually run as the user in the unit file; leaving these commented.
# user = "diegogalindo"
# group = "www-data"

############ Misc ############
# Env variables to pass to the workers (you can also use systemd EnvironmentFile)
raw_env = [
    "DJANGO_SETTINGS_MODULE=config.settings",
    # "SECRET_KEY=supersecret",   # prefer storing secrets outside this file (systemd EnvironmentFile)
]
