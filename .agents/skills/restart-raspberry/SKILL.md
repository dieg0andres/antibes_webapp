---
name: restart-raspberry
description: Start or restart the existing Antibes_webapp Gunicorn application in the webapp tmux session on Raspberry Pi 5 (P1) over SSH. Use for requests to restart the Raspberry Pi web server or recover its stopped application. This restarts the application, not the Raspberry Pi operating system, and does not deploy code.
---

# Restart Raspberry Pi Server

## Purpose and Scope

Start or restart the Antibes_webapp application on Raspberry Pi 5 (P1), using the code already present on the server.

This skill restarts Gunicorn, not the Raspberry Pi operating system. Do not reboot P1, push or pull Git changes, run database migrations, collect static files, or modify web-server configuration. Use the deployment workflow when the user wants to release new code.

When asked to review or edit this skill, only review or edit its files. Execute the procedure only when the user requests an actual application start or restart.

## Environment

- Connect from the MacBook Pro using existing passwordless SSH:

```bash
ssh diegogalindo@pi5_1.antibesln.com
```

- Production application directory:

```text
/home/diegogalindo/my_stuff/01_Projects/antibes_webapp
```

- tmux session: `webapp`.
- Python virtual environment: `.venv` inside the application directory.
- Website: https://app.antibesln.com/
- Stop the foreground application with Ctrl+C.
- User-supplied startup command:

```bash
gunicorn -c gunicorn_config.py config.wsgi:application
```

The deployment skill records a filename discrepancy: the local repository contains `gunicorn_conf.py`, while the supplied command uses `gunicorn_config.py`. Verify the active configuration on P1 from its process arguments and files before stopping anything. If the application is stopped, inspect available configuration and project instructions. If the evidence remains ambiguous, ask the user. Do not rename a file or guess. Record the resolved startup command for this run.

## Procedure

### 1. Connect and Inspect

Connect by SSH. If authentication fails or unexpectedly requests a password, stop and report the issue. Do not change authentication or bypass host verification.

Confirm the application directory exists. Inspect the available Python runtime, virtual environment, Gunicorn configuration, running application processes, and tmux state before changing anything.

Distinguish a missing tmux session from a failed tmux command. If tmux is unavailable or inspection fails, report the problem instead of installing software automatically.

Use tmux inspection commands to locate the exact application pane. For example:

```bash
tmux has-session -t '=webapp'
tmux list-panes -s -t '=webapp' -F '#{session_name}:#{window_index}.#{pane_index} #{pane_id} #{pane_current_path} #{pane_current_command} #{pane_pid} #{pane_dead}'
```

Inspect process arguments and descendants as needed; a pane name or current command alone is not proof of application identity. Record the stable pane ID for subsequent commands. Do not assume pane 0 or the currently selected pane is correct.

Check whether this application's Gunicorn is running outside the intended pane. If so, stop and explain the unexpected process arrangement rather than starting a duplicate or killing unrelated processes.

Where possible, inspect the working website before restarting to establish its expected page content or normal login redirect. If it is already unavailable, proceed with recovery using recognizable application content from the project; do not treat its existing error response as healthy.

### 2. Select or Create the Application Pane

If `webapp` exists:

- Select the verified application pane.
- If Gunicorn is running in its foreground, send Ctrl+C to that pane only.
- Wait up to 30 seconds for the application's master and workers to exit and for the shell to be available.
- If it is already at an idle shell and no application instance is running, skip Ctrl+C.
- If the identified application pane is dead, create a new application window in the same session and record its pane ID; preserve the dead pane for diagnosis.
- If another process is running or the application target is ambiguous, stop and ask for clarification.
- If the application does not stop within the deadline, report the issue. Do not force-kill it or launch another instance.

If `webapp` does not exist:

- Confirm no existing instance of this application is running elsewhere.
- Create a detached session with a shell in the application directory:

```bash
tmux new-session -d -s webapp -c /home/diegogalindo/my_stuff/01_Projects/antibes_webapp
```

- Inspect the new session and record its application pane ID.

The agent may operate on the verified pane through tmux commands over SSH; an interactive attachment is not required. Use the recorded pane ID for every command and inspect output between dependent operations.

### 3. Prepare the Python Environment

In the application pane, change directory:

```bash
cd /home/diegogalindo/my_stuff/01_Projects/antibes_webapp
```

Confirm success before proceeding.

If `.venv` exists and is usable, reuse it. A missing tmux session does not mean the virtual environment is missing.

If `.venv` does not exist:

- Confirm `requirements.txt` exists and the available Python version meets the project's requirements.
- Create the environment:

```bash
python3 -m venv .venv
```

- Confirm creation succeeded.

If `.venv` exists but is broken, stop and report it. Do not delete or overwrite it automatically. If the required Python runtime or venv support is missing, report that setup requirement.

Activate the environment in the same application pane:

```bash
source .venv/bin/activate
```

Confirm activation succeeded and Python resolves inside the application's `.venv`.

For a newly created environment only, install the application's dependencies:

```bash
python -m pip install -r requirements.txt
```

Wait for installation to finish and confirm its exit status. Do not start the application after a failed install. For an existing environment, do not routinely reinstall or upgrade dependencies during a restart. If Gunicorn or another required dependency is missing, report it instead of silently modifying the existing environment.

### 4. Start Gunicorn

In the same pane and activated environment, run the verified startup command. The supplied command is:

```bash
gunicorn -c gunicorn_config.py config.wsgi:application
```

Use the configuration filename resolved during inspection. Confirm that the command starts in the intended application directory.

If attached interactively, detach from tmux, leaving the application running. If working through detached tmux commands, leave the session detached. Do not send `exit` or kill the session.

### 5. Verify the Restart

Allow up to 30 seconds from issuing the startup command, checking periodically with bounded requests:

- Confirm the intended Gunicorn master and workers remain running after detachment.
- Check https://app.antibesln.com/ for the expected application response and recognizable page content, following normal redirects as needed.
- Do not count a proxy error, unrelated page, or the mere presence of a tmux session as success.

If the process exits or the deadline expires, inspect the application pane's startup output and report the failure. If expected website content cannot be established, report process status and explicitly mark website verification as incomplete.

The startup deadline excludes virtual-environment creation and dependency installation.

## Failure Handling and Reporting

Stop at a failed prerequisite or operation. Capture command completion and exit status for directory changes, environment setup, and dependency installation; sending text to tmux is not proof of success. Send each dependent operation only after checking the preceding result.

If SSH disconnects, inspect the actual application and setup state after reconnecting before deciding whether to retry. Do not repeatedly send Ctrl+C or start commands blindly.

Report whether the session was created or reused, whether the environment was created or reused, the verified startup command, application process status, and website verification result. On failure, explain the failed step, any changes already made, and whether the application is stopped. Avoid exposing secrets from configuration or logs.
