RESET = "\033[0m"
BOLD = "\033[1m"

COLOR_INFO = "\033[94m"     # blue
COLOR_QUESTION = "\033[93m" # yellow
COLOR_ANSWER = "\033[92m"   # green

def log_info(message):
    print(f"{COLOR_INFO}{message}{RESET}")