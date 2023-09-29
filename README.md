1. **Backend Setup**:
   - LINK TO BACKEND REPOSITORY: ''
   - DIRECTIONS FOR BACKEND REPOSITORY
    ```bash
    # Clone the repository
    git clone [repo-link] 

    # Set up a virtual environment
    pipenv install
    pipenv shell

    # Start the Flask server
    python app.py
    ```
    At this point, your Flask server should be running on port 8000. In a separate ngrok terminal, type: 'ngrok http 8000'. This will create a forwarding URL for port 8000. The URL that we will fetch from is that '[ngrok-forwarding-url]/classify', which is the endpoint to the server.
