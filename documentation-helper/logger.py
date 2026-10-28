# Color codes for better logging
class Colors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def log_info(message: str, color: str = Colors.CYAN):
    """Log info message with color"""
    print(f"{color}[info] {message}{Colors.END}")


def log_success(message: str):
    """Log success message in green"""
    print(f"{Colors.GREEN}[ok] {message}{Colors.END}")


def log_error(message: str):
    """Log error message in red"""
    print(f"{Colors.RED}[error] {message}{Colors.END}")


def log_warning(message: str):
    """Log warning message in yellow"""
    print(f"{Colors.YELLOW}[warning] {message}{Colors.END}")


def log_header(message: str):
    """Log header message with emphasis"""
    print(f"\n{Colors.BOLD}{Colors.PURPLE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}{message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}{'='*60}{Colors.END}\n")
