#!/usr/bin/env python3
import os
import threading
import logging
from fetch_live_calculate_density import app, initialize_models, background_processor

if __name__ == '__main__':
    try:
        # Initialize models
        initialize_models()
        
        # Start background processing thread
        processor_thread = threading.Thread(target=background_processor, daemon=True)
        processor_thread.start()
        logging.info("Background processor started")
        
        # Get port from environment (Railway sets this)
        port = int(os.environ.get('PORT', 5000))
        
        # Start Flask app
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        exit(1)