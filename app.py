import sys
import logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from datetime import datetime
import traceback
import ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('CMAR_APP')
logger.setLevel(logging.INFO)

try:
    from backend_logic import (
        load_knowledge_base,
        CMARController,
        CSV_FILE_PATH,
        TEXT_MODEL,
        safe_ollama_chat
    )
    logger.info("Successfully imported components from backend_logic.py")
except ImportError as e:
    logger.critical(f"FATAL: Failed to import from backend_logic.py: {e}", exc_info=True)
    print(f"\n--- FATAL SERVER ERROR ---")
    print(f"ERROR: Could not import required components from backend_logic.py.")
    print(f"Please ensure backend_logic.py exists and has no syntax errors.")
    print(f"Error details: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")
    print(f"--- SERVER CANNOT START ---")
    sys.exit(1)
except Exception as e:
     logger.critical(f"FATAL: Unexpected error during import: {e}", exc_info=True)
     print(f"\n--- FATAL SERVER ERROR ---")
     print(f"ERROR: An unexpected error occurred during initial imports.")
     print(f"Error details: {e}")
     print(f"Traceback:\n{traceback.format_exc()}")
     print(f"--- SERVER CANNOT START ---")
     sys.exit(1)

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}, r"/feedback": {"origins": "*"}, r"/health": {"origins": "*"}})
logger.info("Flask app created and CORS enabled for /chat, /feedback, /health.")

controller = None
initialization_error_message = None

try:
    logger.info(f"Attempting to load knowledge base from: {CSV_FILE_PATH}")
    base_df, knowledge_base_dict, df_time_valid = load_knowledge_base(CSV_FILE_PATH)

    if knowledge_base_dict is None or not isinstance(knowledge_base_dict, dict):
        initialization_error_message = "Knowledge base dictionary failed to load or is invalid type. Check CSV structure and logs in backend_logic."
        logger.critical(initialization_error_message)

    else:
        if not knowledge_base_dict:
            logger.warning("Knowledge base dictionary is empty (no valid data in CSV or file empty). Controller will rely solely on LLM.")
        else:
             logger.info(f"Knowledge base loaded successfully with {len(knowledge_base_dict)} categories.")

        logger.info(f"Initializing CMARController with time data validity: {df_time_valid}")
        controller = CMARController(base_df, knowledge_base_dict, df_time_valid)

        if not controller:
            initialization_error_message = "CMARController object failed to initialize unexpectedly after KB load."
            logger.critical(initialization_error_message)
        else:
            logger.info(f"CMARController initialized successfully.")
            if controller.df_time_valid and hasattr(controller, 'time_estimator') and controller.time_estimator.avg_times:
                 logger.info("Time estimation ready: Using historical averages where available, falling back to static rules.")
            elif controller.df_time_valid:
                 logger.warning("Time estimation notice: Historical time data column found and was valid, but no usable average times could be calculated. Using static rules only.")
            else:
                 logger.info("Time estimation ready: Using static rules only (historical time data column missing, invalid, or empty).")


except FileNotFoundError as e:
    initialization_error_message = f"Failed to find necessary file: {e}. Ensure '{CSV_FILE_PATH}' is correct."
    logger.critical(initialization_error_message, exc_info=False)
except ValueError as e:
     initialization_error_message = f"Data error during initialization: {e}. Check CSV format/content or backend_logic parsing."
     logger.critical(initialization_error_message, exc_info=True)
except Exception as e:
    initialization_error_message = f"CRITICAL UNEXPECTED ERROR during controller initialization: {e.__class__.__name__}: {e}"
    logger.critical(initialization_error_message, exc_info=True)

if initialization_error_message:
     print("\n" + "
     print("
     print("
     print(f"CMAR Controller FAILED to initialize properly.")
     print(f"Reason: {initialization_error_message}")
     print(f"Endpoints (/chat, /feedback) might be unavailable or return errors.")
     print("


@app.route('/health', methods=['GET'])
def health_check():
    """Provides a basic health check endpoint."""
    status = {}
    overall_status_code = 200

    if controller and not initialization_error_message:
        status['controller'] = 'OK'
    else:
        status['controller'] = 'Error'
        status['controller_error'] = initialization_error_message or "Controller not initialized."
        overall_status_code = 503

    try:
        ollama.list()
        status['ollama_connection'] = 'OK'
    except Exception as e:
        status['ollama_connection'] = 'Error'
        status['ollama_error'] = f"{e.__class__.__name__}: {e}"
        logger.warning(f"Health Check: Ollama connection test failed: {e}")
        overall_status_code = 503


    status['status'] = 'OK' if overall_status_code == 200 else 'Error'
    return jsonify(status), overall_status_code


@app.route('/chat', methods=['POST', 'OPTIONS'])
def handle_chat():
    """Handles incoming chat messages, routes to controller, returns response."""
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    if request.method == 'POST':
        request_start_time = datetime.now()
        if not controller or initialization_error_message:
            error_msg = f"Service Unavailable: {initialization_error_message or 'Controller not initialized'}"
            logger.error(f"/chat request failed: {error_msg}")
            return jsonify({"error": error_msg}), 503

        if not request.is_json:
            logger.warning("Received non-JSON request to /chat")
            return jsonify({"error": "Invalid request format: Expected JSON."}), 400

        data = request.get_json()
        user_query = data.get('message')
        convo_id = data.get('convo_id')

        if not user_query or not isinstance(user_query, str) or not user_query.strip():
            logger.warning("Received /chat request with missing or invalid 'message' string.")
            return jsonify({"error": "Invalid payload: Missing or empty 'message' string."}), 400

        is_new_convo = False
        if not convo_id:
             is_new_convo = True
             convo_id = f"WEB_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
             logger.info(f"Starting new conversation via /chat: ID '{convo_id}'")
        else:
             logger.info(f"Received /chat message for existing convo: ID '{convo_id}', Query: '{user_query[:100]}...'")

        try:
            # Call the main processing function in the controller
            response_data = controller.process_query(
                convo_id=convo_id,
                query=user_query
            )

            request_end_time = datetime.now()
            duration = (request_end_time - request_start_time).total_seconds()
            logger.info(f"/chat request for Convo ID '{convo_id}' processed successfully in {duration:.3f}s.")


            return jsonify(response_data), 200

        except ValueError as ve:
            error_msg = f"Bad Request Error for convo '{convo_id}': {ve}"
            logger.warning(error_msg)
            return jsonify({"error": str(ve)}), 400
        except ConnectionError as ce:
            error_msg = f"LLM Connection Error for convo '{convo_id}': {ce}"
            logger.error(error_msg, exc_info=False)
            return jsonify({"error": "System Error: Could not connect to the language model service. Please try again later."}), 503
        except RuntimeError as rte:
            error_msg = f"LLM Runtime Error for convo '{convo_id}': {rte}"
            logger.error(error_msg, exc_info=False)
            return jsonify({"error": "System Error: Encountered an issue communicating with the language model."}), 502

        except Exception as e:
            error_msg = f"Unexpected Server Error during /chat processing for Convo ID '{convo_id}': {e.__class__.__name__}: {e}"
            logger.critical(error_msg, exc_info=True)
            return jsonify({"error": "An unexpected internal server error occurred while processing your request."}), 500

@app.route('/feedback', methods=['POST', 'OPTIONS'])
def handle_feedback():
    """Handles yes/no feedback from the user on a proposed solution."""
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    if request.method == 'POST':
        request_start_time = datetime.now()
        if not controller or initialization_error_message:
             error_msg = f"Service Unavailable: {initialization_error_message or 'Controller not initialized'}"
             logger.error(f"/feedback request failed: {error_msg}")
             return jsonify({"error": error_msg}), 503

        if not request.is_json:
            logger.warning("Received non-JSON request to /feedback")
            return jsonify({"error": "Invalid request format: Expected JSON."}), 400

        data = request.get_json()
        convo_id = data.get('convo_id')
        resolved = data.get('resolved')

        if not convo_id:
            logger.warning("Received /feedback request without 'convo_id'.")
            return jsonify({"error": "Invalid payload: Missing 'convo_id'."}), 400
        if resolved is None or not isinstance(resolved, bool):
            logger.warning(f"Received /feedback request for '{convo_id}' with missing or invalid 'resolved' boolean key: Value was '{resolved}'.")
            return jsonify({"error": "Invalid payload: Missing or non-boolean 'resolved' key."}), 400

        logger.info(f"Received /feedback for Convo ID '{convo_id}', Resolved: {resolved}")

        try:
            response_payload = controller.process_feedback(
                convo_id=convo_id,
                success=resolved
            )

            request_end_time = datetime.now()
            duration = (request_end_time - request_start_time).total_seconds()
            logger.info(f"/feedback request for Convo ID '{convo_id}' processed successfully in {duration:.3f}s.")


            return jsonify(response_payload), 200

        except Exception as e:
            error_msg = f"Unexpected Server Error during /feedback processing for Convo ID '{convo_id}': {e.__class__.__name__}: {e}"
            logger.critical(error_msg, exc_info=True)
            return jsonify({"error": "An internal server error occurred while processing your feedback."}), 500


# --- Main Execution ---
if __name__ == '__main__':
    print("\n" + "="*60)
    print("              CMAR Multi-Agent Support System              ")
    print("="*60)

    if initialization_error_message:
         print("--- Initialization Status: FAILED ---")
         print(f"Error: {initialization_error_message}")
         print("WARNING: /chat and /feedback endpoints will likely return errors.")
    elif controller:
        print("--- Initialization Status: SUCCESS ---")
        print(f"Backend Controller: Initialized")
        print(f"LLM Model Configured: {TEXT_MODEL}")
        kb_status = f"Loaded ({len(controller.knowledge_base)} categories)" if controller.knowledge_base else "Empty/Not Loaded"
        print(f"Knowledge Base: {kb_status}")
        time_status = 'Static Rules Only'
        if controller.df_time_valid:
            if hasattr(controller, 'time_estimator') and controller.time_estimator.avg_times:
                time_status = 'Historical Averages + Static Fallback'
            else:
                 time_status = 'Historical Data Valid but No Averages Calculated (Using Static Rules)'
        print(f"Time Estimation Mode: {time_status}")
    else:
         print("--- Initialization Status: UNKNOWN ERROR ---")
         print("ERROR: Controller object is None, but no specific initialization error was recorded.")
         print("WARNING: Endpoints may not function correctly.")

    print("="*60)
    host = '127.0.0.1'
    port = 5000
    print(f"Starting Flask development server...")
    print(f"Listening on http://{host}:{port}")
    print(f"Access the chat interface via frontend.html")
    print(f"API Endpoints:")
    print(f"  - POST http://{host}:{port}/chat")
    print(f"  - POST http://{host}:{port}/feedback")
    print(f"  - GET  http://{host}:{port}/health")
    print("Press CTRL+C to quit.")
    print("="*60 + "\n")

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception as run_error:
        logger.critical(f"Flask server failed to run: {run_error}", exc_info=True)
        print(f"\n--- FATAL SERVER ERROR ---")
        print(f"ERROR: Flask server encountered an error on startup: {run_error}")
        print(f"Check if port {port} is already in use or if there are other system issues.")
        print(f"--- SERVER FAILED TO START ---")
        sys.exit(1)