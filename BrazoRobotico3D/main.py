import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Fallback para X11; cambia a "wayland" si lo necesitas
from render_policy import render_policy_performance

if __name__ == "__main__":
    render_policy_performance()