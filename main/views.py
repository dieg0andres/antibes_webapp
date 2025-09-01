import requests
import pytz

from django.shortcuts import render
from todoist_api_python.api import TodoistAPI
from django.http import JsonResponse
from datetime import datetime, timedelta
from config.settings import TODOIST_TOKEN

headers = {
    "Authorization": f"Bearer {TODOIST_TOKEN}",
}

api = TodoistAPI(TODOIST_TOKEN)

TODOIST_TASKS_URL = "https://api.todoist.com/rest/v2/tasks"
TODOIST_PROJECTS_URL = "https://api.todoist.com/rest/v2/projects"

projects = [
    {"name": "Inbox", "id": "2330914749"},
    {"name": "personal", "id": "2358320501"},
    {"name": "work", "id": "2358320486"},
]

def filter_tasks_by_projects(tasks, projects):
    """
    Filter tasks to only include those associated with the specified projects.
    
    Args:
        tasks (list): List of task dictionaries from Todoist API
        projects (list): List of project dictionaries with 'id' keys
    
    Returns:
        list: Filtered list of tasks that belong to the specified projects
    """
    # Extract project IDs from the projects list
    project_ids = {project['id'] for project in projects}
    
    # Filter tasks that have project_id in our project_ids set
    filtered_tasks = [task for task in tasks if task.get('project_id') in project_ids]
    
    return filtered_tasks

def convert_priority(priority):
    """
    Convert Todoist priority numbers to P1-P4 format.
    
    Args:
        priority (int): Todoist priority (1-4, where 1 is highest)
    
    Returns:
        str: Priority in P1-P4 format (where P1 is highest)
    """
    priority_mapping = {
        1: "P4",  # Todoist priority 1 = P4 (lowest)
        2: "P3",  # Todoist priority 2 = P3
        3: "P2",  # Todoist priority 3 = P2
        4: "P1",  # Todoist priority 4 = P1 (highest)
    }
    return priority_mapping.get(priority, "P4")  # Default to P4 if priority is not 1-4

def update_task_priorities(tasks):
    """
    Update the priority field in each task dictionary to use P1-P4 format.
    
    Args:
        tasks (list): List of task dictionaries
    
    Returns:
        list: List of task dictionaries with updated priority fields
    """
    for task in tasks:
        # Convert the numeric priority to P1-P4 format and update the task
        task['priority'] = convert_priority(task['priority'])
    
    return tasks

def get_today_date():
    """
    Get today's date in the local timezone.
    """
    # Use UTC time and convert to Central Time (CDT/CST)
    utc_now = datetime.now(pytz.UTC)
    central_tz = pytz.timezone('America/Chicago')  # Central Time
    local_time = utc_now.astimezone(central_tz)
    return local_time.date()

def print_tasks_due_today(filtered_tasks):
    """
    Print tasks that are due today from the filtered_tasks list.
    
    Args:
        filtered_tasks (list): List of filtered task dictionaries
    """
    today = get_today_date()
    print(f"\n=== TASKS DUE TODAY ({today}) ===")
    
    tasks_due_today = []
    
    for task in filtered_tasks:
        due_field = task.get('due')
        if due_field is not None:
            try:
                # Extract the date string from the due field
                if isinstance(due_field, dict) and 'date' in due_field:
                    due_date_str = due_field['date']
                elif isinstance(due_field, str):
                    due_date_str = due_field
                else:
                    continue
                
                # Parse the due date
                due_date = datetime.strptime(due_date_str.split('T')[0], '%Y-%m-%d').date()
                
                # Check if due today
                if due_date == today:
                    tasks_due_today.append(task)
                    
            except (ValueError, KeyError, AttributeError) as e:
                print(f"Error parsing due date for task {task.get('id', 'unknown')}: {e}")
                continue
    
    if tasks_due_today:
        print(f"Found {len(tasks_due_today)} tasks due today:")
        for i, task in enumerate(tasks_due_today, 1):
            project_name = next((p['name'] for p in projects if p['id'] == task['project_id']), 'Unknown')
            print(f"\n{i}. Task: {task['content']}")
            print(f"   Project: {project_name}")
            print(f"   Priority: {task['priority']}")
            print(f"   Due: {task['due']}")
            print(f"   ID: {task['id']}")
    else:
        print("No tasks found due today.")
    
    print("=" * 50)
    return tasks_due_today

def create_dashboard_data(filtered_tasks):
    """
    Create dashboard data structure with task counts by priority for the next 7 days.
    
    Args:
        filtered_tasks (list): List of filtered task dictionaries
    
    Returns:
        dict: Dashboard data with work and personal project data
    """
    # Get project IDs for work and personal
    work_project_id = next((p['id'] for p in projects if p['name'] == 'work'), None)
    personal_project_id = next((p['id'] for p in projects if p['name'] == 'personal'), None)
    
    # Generate next 7 days of dates (starting from today)
    today = get_today_date()
    
    # Create separate date lists for each project to avoid sharing references
    def create_dates_list():
        dates = []
        for i in range(10):  # Changed from 7 to 10 days
            date = today + timedelta(days=i)
            dates.append({
                'date': date.strftime('%Y-%m-%d'),  # Convert to string for JSON serialization
                'day_name': date.strftime('%A'),
                'date_str': date.strftime('%Y-%m-%d'),
                'display_date': date.strftime('%b %d')
            })
        return dates
    
    # Initialize dashboard structure with separate date lists
    dashboard_data = {
        'work': {'dates': create_dates_list(), 'project_name': 'Work'},
        'personal': {'dates': create_dates_list(), 'project_name': 'Personal'}
    }
    
    # Process tasks for each project
    for project_type, project_id in [('work', work_project_id), ('personal', personal_project_id)]:
        if project_id:
            # Filter tasks for this specific project
            project_tasks = [task for task in filtered_tasks if task['project_id'] == project_id]
            
            # Initialize counts for each date
            for date_data in dashboard_data[project_type]['dates']:
                date_data['counts'] = {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'total': 0}
            
            # Count tasks by priority for each date
            for task in project_tasks:
                # Check if task has a due date
                due_field = task.get('due')
                if due_field is not None:
                    try:
                        # Extract the date string from the due field
                        # Handle the structure: {'date': '2025-09-13', 'is_recurring': False, 'lang': 'en', 'string': '13 Sep'}
                        if isinstance(due_field, dict) and 'date' in due_field:
                            due_date_str = due_field['date']
                        elif isinstance(due_field, str):
                            due_date_str = due_field
                        else:
                            continue  # Skip if due field format is unexpected
                        
                        # Parse the due date (Todoist format: "2025-09-13")
                        due_date = datetime.strptime(due_date_str.split('T')[0], '%Y-%m-%d').date()
                        
                        # Skip overdue tasks (due date before today)
                        if due_date < today:
                            continue
                        
                        # Convert to string for comparison
                        due_date_str = due_date.strftime('%Y-%m-%d')
                        
                        # Find the corresponding date in our dashboard
                        for date_data in dashboard_data[project_type]['dates']:
                            if date_data['date'] == due_date_str:
                                priority = task['priority']
                                if priority in date_data['counts']:
                                    date_data['counts'][priority] += 1
                                    date_data['counts']['total'] += 1
                                break
                    except (ValueError, KeyError, AttributeError) as e:
                        # If there's an issue parsing the date, skip this task
                        continue
    
    return dashboard_data

# Create your views here.
def index(request):
    return render(request, 'main/index.html')


def todoist(request):
    response = requests.get(TODOIST_TASKS_URL, headers=headers)
    tasks = response.json()
    
    # Filter tasks to only show those in our defined projects
    filtered_tasks = filter_tasks_by_projects(tasks, projects)
    
    # Update the priority field in the filtered tasks
    filtered_tasks = update_task_priorities(filtered_tasks)
    
    # Create dashboard data structure
    dashboard_data = create_dashboard_data(filtered_tasks)

    return render(request, 'main/todoist.html', {'dashboard_data': dashboard_data})