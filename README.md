### Make sure venv is active
source ~/.venv/bin/activate

### Recreating the environment elsewhere
On another machine or new virtual environment:

  python -m venv ~/.venv_new
  source ~/.venv_new/bin/activate
  pip install -r requirements.txt
