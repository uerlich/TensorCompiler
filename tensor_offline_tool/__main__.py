from .tool.main import run_cli, run_gui
import sys
if len(sys.argv) > 1:
    run_cli()
else:
    run_gui()
