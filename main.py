#!/usr/bin/env python3
import os
import threading
import logging
import traceback
from fetch_live_calculate_density import app, initialize_models, background_processor

if __name__ == '__main__':
    # Ensure TensorFlow optimizations are disabled
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    try:
        # Initialize models (now always loads)
        logging.info("Starting model initialization")
        initialize_models()
        
        # Always start background processing thread
        processor_thread = threading.Thread(target=background_processor, daemon=True)
        processor_thread.start()
        logging.info("Background processor started")
        
        # Get port from environment (Railway sets this)
        port = int(os.environ.get('PORT', 5000))
        
        # Start Flask app
        logging.info(f"Starting Flask app on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        exit(1)