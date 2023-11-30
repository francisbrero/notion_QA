# notion_QA
Build a chatbot trained on your Notion Workspace data

# Download your Notion Workspace data
1. Go to your Notion Workspace
2. Click on Settings & Members
3. Click on Settings
4. Click on Export Content
5. Click on Export
6. Wait for the email from Notion
7. Download the zip file
8. Unzip the file
9. Rename the folder to "notion_data"

NB: you may also download the data from one sub page of your Notion Workspace to try things out

# Technical setup
## Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

## Install the requirements
```
pip install -r requirements.txt
```

# Run embeddings for a Notion workspace
```bash
python ./notion/train.py --n support_runbook
```