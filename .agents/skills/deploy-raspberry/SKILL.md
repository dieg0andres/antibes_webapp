---
name: deploy-raspberry
description: Deploy Antibes_webapp from the MacBook Pro to Raspberry Pi 5 (P1) through GitHub and SSH, install Python requirements, collect static files, restart Gunicorn in tmux, and verify the website. Use when the user requests deployment of Antibes_webapp to P1.
---

# Deploy Antibes_webapp to Raspberry Pi (P1)

## Purpose

Deploy the latest committed version of Antibes_webapp from the MacBook Pro to production server Raspberry Pi 5 (P1).

Follow this sequence: confirm database requirements, validate the development repository, inspect production, push to GitHub, validate and update the production repository, install Python dependencies, collect static files, restart the application in tmux, and verify the website.

## Environment

### Development

- Machine: MacBook Pro.
- Repository and application working directory:

```text
/Users/diegogalindo/my_stuff/01_Projects/antibes_webapp
```

### Production

- Machine: Raspberry Pi 5 (P1).
- Repository and application working directory:

```text
/home/diegogalindo/my_stuff/01_Projects/antibes_webapp
```

- Connection method: SSH using existing passwordless authentication.

```bash
ssh diegogalindo@pi5_1.antibesln.com
```

- tmux session: `webapp`.
- Application stopping method: Ctrl+C in the application's tmux pane.
- Website: https://app.antibesln.com/

### Shared Configuration

- Required Git branch: `master`.
- Required Git remote name: `github`.
- Expected GitHub repository: https://github.com/dieg0andres/antibes_webapp
- Python virtual environment activation, from the respective application directory:

```bash
source .venv/bin/activate
```

Equivalent HTTPS or SSH remote URLs are acceptable if they identify the expected GitHub repository.

Production dependency installation command:

```bash
python -m pip install -r requirements.txt
```

Production static-file collection command:

```bash
python manage.py collectstatic --noinput
```

User-supplied production startup command:

```bash
gunicorn -c gunicorn_config.py config.wsgi:application
```

### Initial Setup Checks

- Identify the application window and pane within `webapp` by inspecting the running process and working directory.
- Establish the website's expected successful response and recognizable page content, plus a representative CSS or JavaScript URL to verify.
- Confirm production serves `/static/` from its configured static-file location. The local settings define `STATIC_ROOT` as `BASE_DIR / 'staticfiles'`; verify the production configuration. Collection alone does not configure static-file serving.
- Resolve the Gunicorn configuration filename before deployment. The local repository was observed to contain `gunicorn_conf.py`, while the user supplied `gunicorn_config.py`. Inspect P1's existing process and files to identify the active configuration. If the evidence does not resolve the mismatch, ask the user. Do not rename files or silently guess. Record the verified startup command and use it below.

## Deployment Boundaries

Both repositories must be on `master` with clean working trees before their respective Git updates.

Treat staged changes, unstaged changes, and untracked files as an unclean working tree.

If either repository is not on `master` or has changes, stop and inform the user. Do not automatically switch branches, commit, stash, discard files, or resolve conflicts.

Verify that a remote named `github` exists on each machine and points to the expected repository. Stop and inform the user if it is missing or incorrect. Do not create or change remotes automatically.

Always explicitly push and pull `master` using the `github` remote.

Do not force-push or overwrite production changes.

This workflow includes Python dependency installation and static-file collection. It does not include database changes or migrations. If required release steps fall outside this workflow, stop and report them.

## Deployment Procedure

### 1. Confirm Database Requirements

Before making deployment changes, ask the user:

“Does this deployment require any database changes or migrations?”

- If yes, stop and explain that those steps must be defined before deployment.
- If no, proceed.
- If the answer is uncertain, stop until clarified.

An explicit answer already provided for this deployment is sufficient; do not ask again.

Do not run database changes or migration commands as part of this skill.

### 2. Validate the Development Repository

On the MacBook Pro, navigate to:

```bash
cd /Users/diegogalindo/my_stuff/01_Projects/antibes_webapp
```

- Confirm this is the intended development repository.
- Confirm the current branch is `master`.
- Confirm the working tree is clean, with nothing to commit.
- Confirm `requirements.txt` exists and is tracked by Git.
- Record the commit intended for deployment.

If any check fails, stop and inform the user.

Activate the development Python environment before running any Python application commands:

```bash
source .venv/bin/activate
```

### 3. Inspect Production and Establish Verification

Connect using:

```bash
ssh diegogalindo@pi5_1.antibesln.com
```

Use the existing passwordless authentication. If the connection fails or unexpectedly requires a password, stop and report the issue. Do not change authentication settings or bypass SSH host verification.

Before changing production:

- Complete the initial setup checks above.
- Inspect the `webapp` tmux session if it exists.
- Identify the application window and pane from their working directory and running process.
- Record the exact tmux target for this deployment.
- Do not assume the currently selected pane is the application pane.
- If the target is ambiguous, ask the user before sending commands.

Inspect https://app.antibesln.com/ and establish the expected successful response and recognizable application page content. Follow normal redirects if appropriate. Identify a representative CSS or JavaScript file for verification.

Do not treat an error page as the expected healthy response. If the website is unavailable or its expected behavior is unclear, ask the user to establish verification criteria before proceeding.

### 4. Push to GitHub

In the development repository:

- Reconfirm that the branch is `master`, the working tree is clean, and the commit has not changed since validation.
- Verify that the remote named `github` exists and points to `dieg0andres/antibes_webapp`.
- Run:

```bash
git push github master
```

- Confirm the push succeeded and the remote `master` branch matches the intended deployment commit.

If validation or the push fails, stop and inform the user before updating P1.

### 5. Validate the Production Repository

On P1, navigate to:

```bash
cd /home/diegogalindo/my_stuff/01_Projects/antibes_webapp
```

- Confirm this is the intended production repository.
- Confirm the current branch is `master`.
- Confirm the working tree is clean, with nothing to commit.
- Record the current production commit.
- Verify that the remote named `github` exists and points to `dieg0andres/antibes_webapp`.

If any check fails, stop and inform the user before pulling.

### 6. Update the Production Repository

Run:

```bash
git pull github master
```

Do not add a fast-forward-only requirement or change Git's configured pull behavior.

- Confirm the pull completed successfully.
- Confirm the working tree remains clean.
- Confirm the production commit matches the intended deployment commit.
- Confirm `requirements.txt`, `.venv/bin/activate`, `manage.py`, and the verified Gunicorn configuration file exist.

If a check fails, stop and inform the user before stopping the application. Do not automatically resolve divergence or conflicts.

### 7. Prepare the Application's tmux Pane

Check whether the tmux session named `webapp` exists.

If the session exists:

- Access the application window and pane identified during inspection.
- Reconfirm the target before sending commands.
- If the application is running in the foreground, send Ctrl+C to stop it.
- Wait for the application process to stop and the shell prompt to return.
- If the application is already stopped and the pane is at a shell prompt, skip Ctrl+C.
- If another process is running or the pane is ambiguous, stop and report the issue.
- If the application does not stop within 30 seconds, report the issue rather than starting a second instance or forcing termination.

If the session does not exist:

- Create a tmux session named `webapp`.
- Access its application shell and record its window and pane target.

### 8. Activate the Environment and Install Dependencies

In the application's tmux pane, run:

```bash
cd /home/diegogalindo/my_stuff/01_Projects/antibes_webapp
source .venv/bin/activate
```

Confirm that changing directories and activating the environment both succeed. Confirm that the selected Python belongs to the application's `.venv`.

Activation must occur in the same shell used for dependency installation, static-file collection, and Gunicorn startup. Activation in a separate SSH shell does not activate an existing tmux pane.

Confirm `requirements.txt` exists, then run:

```bash
python -m pip install -r requirements.txt
```

Wait for installation to finish and confirm a successful exit status before proceeding.

If installation fails:

- Stop and report the installation error.
- Clearly state that the application has been stopped and dependencies may have changed partially.
- Do not start the application or attempt automatic rollback.

Do not automatically upgrade pip, recreate the virtual environment, or change dependency versions beyond what `requirements.txt` specifies.

### 9. Collect Static Files

After successfully installing dependencies, run in the same tmux pane, application directory, and activated virtual environment:

```bash
python manage.py collectstatic --noinput
```

Wait for completion and confirm a successful exit status.

If collection fails, stop and report the error. State that the application remains stopped and collected files may have changed partially. Do not proceed with startup.

Do not add `--clear` or alter the web-server configuration as part of this step.

### 10. Start the Application

In the same tmux pane and activated virtual environment, run the startup command verified during initial inspection. The user-supplied command is:

```bash
gunicorn -c gunicorn_config.py config.wsgi:application
```

Use this exact command only if that configuration filename has been verified. Otherwise use the resolved command recorded during initial setup.

Detach from tmux, leaving the session and application running. Do not terminate the session or send `exit` to the application pane.

### 11. Verify the Deployment

Starting when the Gunicorn startup command is issued:

- Allow up to 30 seconds for startup, checking periodically.
- Confirm the application process remains running after detachment.
- Check https://app.antibesln.com/ against the expected response and recognizable application page content.
- Confirm the representative CSS or JavaScript URL returns the expected asset, not an HTML error or login page.
- Use bounded requests so website checks do not wait indefinitely.
- If verification fails or the deadline expires, inspect startup output and report the failure.

The startup deadline does not apply to dependency installation or static-file collection.

Do not report deployment as successful merely because the startup command was sent, the tmux session exists, or an unrelated page responds.

## Failure Handling

Stop at the first failed prerequisite or deployment step.

Report:

- Which step failed.
- The observed error or unexpected state.
- Which changes already occurred.
- Whether the application's availability was verified.
- What is needed to proceed.

If an SSH connection drops during an operation, inspect the actual repository, installation, static-file collection, and application state after reconnecting before deciding whether to retry.

Do not retry operations that change production state blindly. Automatic rollback is not defined in this first version.

## Completion Report

Report:

- Deployed commit.
- Production update result.
- Dependency installation result.
- Static-file collection result.
- tmux restart result.
- Website and static-asset verification results.

Clearly distinguish a verified successful deployment from an incomplete or failed deployment.
