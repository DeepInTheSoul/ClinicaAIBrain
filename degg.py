import os
import sys
import subprocess
# get python path
python_path = os.path.abspath(sys.executable)

# get streamlit path
script_streamlit = os.path.join(os.path.dirname(python_path), r'streamlit')

# my script
current_dir = os.path.dirname(os.path.abspath(__file__))
script_webserver = os.path.join(current_dir, r'app.py')

# Here is the magic to execute  "streamlit run WebServer.py"
# Execute "streamlit run app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", script_webserver])