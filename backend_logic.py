import pandas as pd
import ollama
from datetime import datetime
import sys
import logging
import re
import numpy as np

# *** IMPORTANT: UPDATE THIS PATH IF YOUR CSV IS NOT IN THE SAME FOLDER ***
CSV_FILE_PATH = r'/Users/shourjyaghosh/Downloads/Dataset/[Usecase 7] AI-Driven Customer Support Enhancing Efficiency Through Multiagents​/Historical_ticket_data.csv'
TEXT_MODEL = 'llama3'
# --- FIX 1: Reduce history turns ---
HISTORY_TURNS = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - AGENT - %(message)s', stream=sys.stdout)


def load_knowledge_base(csv_path):
    """Loads and prepares the knowledge base and historical data from the CSV. Returns df, knowledge_base_dict, df_time_valid"""
    df_time_valid = False
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        logging.info(f"Loaded CSV columns: {df.columns.tolist()}")

        required_cols = ['Issue Category', 'Solution', 'Priority']
        time_col = 'Resolution Time (hours)'

        missing_core_cols = [col for col in required_cols if col not in df.columns]
        if missing_core_cols:
            error_msg = f"CSV Missing required core columns: {missing_core_cols}. KB cannot be built."
            logging.error(error_msg)
            return pd.DataFrame(columns=df.columns), {}, False

        if time_col in df.columns:
            original_non_nan_count = df[time_col].notna().sum()
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            coerced_nan_count = df[time_col].isna().sum()
            converted_to_nan = coerced_nan_count - (len(df) - original_non_nan_count)

            if converted_to_nan > 0:
                 logging.warning(f"Found {converted_to_nan} non-numeric entries in '{time_col}'. Converted to NaN.")

            rows_before_dropna = len(df)
            df.dropna(subset=[time_col], inplace=True)
            rows_dropped = rows_before_dropna - len(df)

            if rows_dropped > 0:
                 logging.info(f"Excluded {rows_dropped} rows due to missing or invalid '{time_col}' values.")

            if not df.empty:
                 logging.info(f"Processed numeric data in '{time_col}'. {len(df)} rows usable for time estimation.")
                 df_time_valid = True
            else:
                 logging.warning(f"No rows remaining after processing '{time_col}'. Time data is invalid or all rows had issues.")
                 df_time_valid = False

        else:
            logging.warning(f"Optional column '{time_col}' not found. Time estimation will use static defaults.")
            if time_col not in df.columns:
                 df[time_col] = np.nan
            df_time_valid = False

        initial_rows = len(df)
        nan_issue_cat = df['Issue Category'].isna().sum()
        nan_solution = df['Solution'].isna().sum()
        if nan_issue_cat > 0: logging.warning(f"Found {nan_issue_cat} rows with missing 'Issue Category' (NaN).")
        if nan_solution > 0: logging.warning(f"Found {nan_solution} rows with missing 'Solution' (NaN).")

        df.dropna(subset=['Issue Category', 'Solution'], inplace=True)
        rows_after_dropna = len(df)
        if initial_rows > rows_after_dropna:
            logging.info(f"Dropped {initial_rows - rows_after_dropna} rows due to missing 'Issue Category' or 'Solution' (NaN values).")

        df['Issue Category'] = df['Issue Category'].astype(str).str.strip()
        df['Solution'] = df['Solution'].astype(str).str.strip()
        df['Priority'] = df['Priority'].fillna('Medium').astype(str).str.strip()


        initial_rows_before_empty_filter = len(df)
        df = df[(df['Issue Category'] != '') & (df['Solution'] != '')]
        rows_after_empty_filter = len(df)
        if rows_after_empty_filter < initial_rows_before_empty_filter:
             logging.info(f"Dropped {initial_rows_before_empty_filter - rows_after_empty_filter} rows due to empty 'Issue Category' or 'Solution' after stripping.")

        if df.empty:
            logging.error("No valid KB data remained after cleaning (NaNs, empty strings). KB will be empty.")
            return df, {}, df_time_valid

        knowledge_base = df.groupby('Issue Category')['Solution'].apply(
            lambda x: list(set(s for s in x if s))
        ).to_dict()

        logging.info(f"KB built: {len(knowledge_base)} categories. DataFrame rows available for analysis: {len(df)}. Time data valid: {df_time_valid}.")
        return df, knowledge_base, df_time_valid

    except FileNotFoundError:
        logging.error(f"FATAL: CSV file not found at the specified path: {csv_path}")
        raise FileNotFoundError(f"Required data file not found: {csv_path}")
    except pd.errors.EmptyDataError:
        logging.error(f"FATAL: CSV file is empty: {csv_path}")
        raise ValueError(f"Required data file is empty: {csv_path}")
    except Exception as e:
        logging.error(f"FATAL: Unexpected error loading/parsing CSV '{csv_path}': {e}", exc_info=True)
        raise


def safe_ollama_chat(messages: list, model: str = TEXT_MODEL) -> str:
    """Safely interacts with Ollama LLM using message history format."""
    prompt_text_log = "\n".join([f"{m['role']}: {m['content'][:100]}{'...' if len(m['content']) > 100 else ''}" for m in messages])
    logging.debug(f"Sending {len(messages)} messages to Ollama ({model}). Context preview:\n{prompt_text_log}")

    MAX_PROMPT_CHARS_APPROX = 4000
    prompt_char_len = sum(len(m['content']) for m in messages)
    if prompt_char_len > MAX_PROMPT_CHARS_APPROX:
        logging.warning(f"Combined message context approx {prompt_char_len} chars > {MAX_PROMPT_CHARS_APPROX}. Potential Ollama truncation/performance issues.")

    try:
        if not isinstance(messages, list) or not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
             logging.error(f"Invalid 'messages' format passed to safe_ollama_chat: {type(messages)}")
             raise ValueError("Invalid format for messages argument in safe_ollama_chat. Expected list of dicts.")

        response = ollama.chat(model=model, messages=messages)

        if not response or 'message' not in response or 'content' not in response['message']:
            logging.error(f"Ollama returned an unexpected or incomplete response structure: {response}")
            raise RuntimeError("Ollama returned an invalid response structure.")

        content = response['message']['content'].strip()
        logging.debug(f"Ollama response received ({len(content)} chars).")

        if not content:
            logging.warning(f"Ollama returned an empty content string for model {model}. Review prompt and context:\n{prompt_text_log}")
            raise RuntimeError("Ollama returned empty content.")
        return content

    except Exception as e:
        e_str = str(e).lower()
        if ('connection refused' in e_str):
            logging.error(f"Ollama connection error: Connection refused. Is Ollama service running and accessible? Details: {e}")
            raise ConnectionError(f"Ollama connection refused. Service might be down or network issue. Error: {e}")
        elif ('model' in e_str and ('not found' in e_str or 'pull model' in e_str)):
            logging.error(f"Ollama model error: Model '{model}' not found or needs pulling. Run `ollama pull {model}`. Details: {e}")
            raise ConnectionError(f"Ollama model '{model}' not found. Ensure it is pulled/available. Error: {e}")
        elif isinstance(e, (ConnectionError, RuntimeError, ValueError)):
             raise e
        else:
            logging.error(f"Ollama API call failed unexpectedly for model {model}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama API call failed unexpectedly: {e}")


def is_query_vague(query: str) -> bool:
    """Checks if a query describing a problem is likely too vague, especially on the first turn."""
    query_lower = query.lower().strip()
    if not query_lower: return True

    # Common problem-related keywords
    problem_keywords = ['error', 'issue', 'problem', 'bug', 'fail', 'crash', 'won\'t', 'doesn\'t', 'not working', 'broken', 'stuck', 'frozen', 'unable', 'cannot', 'sync', 'payment', 'login', 'install', 'update', 'connect', 'slow']

    # Check if the query is short AND lacks specific problem keywords
    if len(query.split()) < 5 and not any(keyword in query_lower for keyword in problem_keywords):
        if query_lower not in ['yes', 'no', 'y', 'n'] and 'work' not in query_lower:
            logging.debug(f"Query deemed potentially vague (short & lacks clear problem keywords): '{query}'")
            return True

    return False


def is_informational_query(query: str) -> bool:
    """Uses keywords and LLM confirmation to check if query primarily seeks information."""
    query_lower = query.lower().strip()
    if not query_lower: return False

    info_patterns = [
        'how do i', 'how to', 'where is', 'what is', 'can i contact', 'contact info',
        'phone number for', 'email for', 'website for', 'link for', 'steps to',
        'instructions for', 'explain', 'tell me about', 'what are the specs',
        'user guide', 'manual for', 'feature list', 'comparison', 'warranty'
    ]
    # Keywords strongly suggesting information request
    info_keywords = ['contact support', 'support number', 'help page', 'documentation', 'faq', 'knowledge base']

    is_info_pattern = any(query_lower.startswith(p) for p in info_patterns) or \
                      any(kw in query_lower for kw in info_keywords)

    if is_info_pattern:
        logging.info(f"Keyword pattern suggests informational query: '{query[:100]}...'")
        # Use LLM to confirm, as keywords can sometimes appear in problem descriptions
        prompt = f"""Does the following user query primarily ask for general information, instructions, contact details, links, explanations, or product features, rather than describing a specific technical problem, error, or malfunction they are personally experiencing right now?

User Query: "{query}"

Answer ONLY with 'Information Request' or 'Problem Description'.
"""
        try:
            messages = [{'role': 'user', 'content': prompt}]
            response = safe_ollama_chat(messages)
            if "information request" in response.lower():
                logging.info("LLM confirms informational query.")
                return True
            else:
                logging.info("LLM classifies as problem description or other.")
                return False
        except Exception as e:
            # If LLM check fails, rely cautiously on keyword match but log warning
            logging.warning(f"LLM info check failed: {e}. Relying on keyword match for informational query status.")
            return True # Or False, depending on desired fallback behavior. Let's stick with True if keywords matched.
    return False



class SummaryAgent:
    """Generates a concise summary of the customer's query intent (for internal logs)."""
    def generate_summary(self, query: str) -> str:
        """Generates a brief summary of the user's core issue or request."""
        if not query: return "N/A (Empty Query)"
        prompt = f"Summarize the core user need or problem from the following query in one concise sentence (max 15 words). Focus on the main issue or question. Query: \"{query}\""
        try:
            messages = [{'role': 'user', 'content': prompt}]
            summary = safe_ollama_chat(messages, model=TEXT_MODEL).replace('"', '').strip()
            return summary or "Summary generation failed (empty LLM response)."
        except Exception as e:
            logging.error(f"SummaryAgent Error: {e}.")
            return f"Summary generation failed: {e}"


class ActionExtractor:
     """Maps the internal state/decision to a loggable description of the agent's action."""
     def extract_actions(self, query: str, recommended_action_type: str) -> str:
        """Provides a human-readable log string for the agent's action."""
        action_map = {
            "ASK_CLARIFICATION": "Agent asking user for more details/clarification.",
            "PROVIDE_SOLUTION": "Agent suggesting a specific troubleshooting step.",
            "PROVIDE_INFO": "Agent providing requested information/answer.",
            "GREETING": "Agent responded to user greeting.",
            "AWAITING_FEEDBACK": "Agent provided a step and is waiting for user feedback (Yes/No).",
            "FEEDBACK_POSITIVE": "Agent processing positive feedback ('Yes').",
            "FEEDBACK_NEGATIVE": "Agent processing negative feedback ('No').",
            "OUT_OF_SCOPE": "Agent identified query as out of scope/unrelated.",
            "ERROR_LLM_CONNECTION": "Agent experienced LLM/Connection error.",
            "ERROR_INVALID_INPUT": "Agent detected invalid user input.",
            "ERROR_UNEXPECTED": "Agent experienced an unexpected internal error.",
            "INTERNAL_ERROR_STATE": "Agent reached an unexpected internal state.",
        }
        default_text = f"Agent action type '{recommended_action_type}' - needs review if unexpected."
        return action_map.get(recommended_action_type, default_text)


class InitialTroubleshooter:
    """Suggests steps, asks questions, or answers info queries, using KB, history, and LLM reasoning."""
    def __init__(self, knowledge_base: dict):
        self.knowledge_base = knowledge_base if knowledge_base else {}
        logging.info(f"InitialTroubleshooter initialized with {len(self.knowledge_base)} KB categories.")

    def suggest_initial_steps(self, category: str, query: str, conversation_history: list) -> tuple[str, str]:
        """
        Provides the next response (step, question, or info) based on query, history, and KB.
        Returns: (response_text, action_type)
        Action Types: PROVIDE_SOLUTION, ASK_CLARIFICATION, PROVIDE_INFO
        """
        current_turn_message = {'role': 'user', 'content': query}
        history_log = []
        if conversation_history:
            for turn in conversation_history:
                role = turn.get('role', 'unknown').capitalize()
                content = turn.get('content', '')
                history_log.append(f"{role}: {content[:100]}{'...' if len(content) > 100 else ''}")
            history_str_log = "\n".join(history_log)
        else:
            history_str_log = "No previous conversation history."


        logging.debug(f"Troubleshooter assessing: Category='{category}'. History context (for LLM):\n{history_str_log}")

        if is_informational_query(query):
            logging.info(f"Handling informational query: '{query[:100]}...'")
            prompt_messages = conversation_history + [
                {'role': 'system', 'content': "You are a helpful information assistant. The user is asking for information (not describing a technical problem). Based on the conversation history and their latest question, provide a concise and accurate answer. If the request is ambiguous or you lack the specific information, ask ONE single, crucial clarifying question to understand what they need. Do not suggest troubleshooting steps."},
                current_turn_message
            ]
            try:
                llm_answer = safe_ollama_chat(prompt_messages)
                is_question = '?' in llm_answer and len(llm_answer.split()) > 4 and any(kw in llm_answer.lower() for kw in ['what', 'which', 'specify', 'provide', 'could you', 'can you', 'do you mean', 'clarify'])
                if is_question:
                    logging.info(f"LLM asks for clarification on info request: '{llm_answer}'")
                    return llm_answer.strip(), "ASK_CLARIFICATION"
                else:
                    llm_answer_cleaned = re.sub(r'^(Okay|Sure|Certainly|Here is|The info is|You asked about|Based on.*?,?)\s*', '', llm_answer, flags=re.IGNORECASE).strip()
                    if not llm_answer_cleaned:
                         logging.warning("LLM provided an empty informational response after cleaning. Asking for clarification.")
                         return "I couldn't retrieve that specific information right now. Could you please rephrase your request or specify what you need?", "ASK_CLARIFICATION"
                    logging.info(f"LLM provided info: '{llm_answer_cleaned[:150]}...'")
                    return llm_answer_cleaned, "PROVIDE_INFO"
            except Exception as e:
                logging.error(f"LLM error during informational query handling: {e}.")
                return "I can try to help with that. Could you please specify exactly what information you need?", "ASK_CLARIFICATION"

        if not conversation_history and is_query_vague(query):
            logging.info("Handling vague first query without history.")
            clarification_prompt = ("I understand you're having an issue, but I need a little more detail to help effectively. Could you please describe:\n"
                                   "- What exactly are you trying to do?\n"
                                   "- What happens when you try (including any error messages)?\n"
                                   "- When did this problem start?")
            return clarification_prompt, "ASK_CLARIFICATION"

        logging.info(f"Handling problem description/follow-up. Category: '{category or 'Unknown'}'. Query: '{query[:100]}...'")

        kb_solutions_prompt_part = ""
        category_to_use = category if category and category != "General Inquiry" else None
        if category_to_use and category_to_use in self.knowledge_base:
            solutions = self.knowledge_base.get(category_to_use, [])
            if solutions:
                MAX_KB_EXAMPLES = 3
                solution_examples = "\n - ".join(solutions[:MAX_KB_EXAMPLES])
                kb_solutions_prompt_part = f"\n\nHints from knowledge base for '{category_to_use}':\n - {solution_examples}"
                logging.debug(f"Providing {len(solutions[:MAX_KB_EXAMPLES])} KB examples for '{category_to_use}' to LLM.")

        # --- FIX 2: Simplified and Refocused System Prompt ---
        system_prompt = f"""You are an expert technical support agent. The user is reporting a problem related to '{category_to_use or 'Unknown Issue'}'.
Review the conversation history provided in the messages prior to this one, and the user's latest message below.

User's Latest Message: "{query}"
{kb_solutions_prompt_part}

**Your Task: Determine the SINGLE BEST next action.**

**Choose ONE option:**
1.  **Suggest ONE Actionable Step:** If possible, suggest ONE simple, concrete troubleshooting step relevant to the user's latest message and the history.
    *   **CRITICAL: DO NOT REPEAT** steps already mentioned in the history.
    *   Example: "Try force restarting your device."
2.  **Ask ONE Specific Question:** If you MUST have more information to diagnose the issue, ask ONE specific, targeted question.
    *   **CRITICAL: DO NOT ASK GENERIC QUESTIONS** like "Tell me more?". Ask for *new*, *specific* details needed for the next step.
    *   Example: "What error message number do you see on the screen?"

**Output:** Respond ONLY with the suggested step OR the specific question. Be concise. Do not add conversational filler, summaries, or introductions unless the step/question itself requires it.
"""

        prompt_messages = conversation_history + [
             {'role': 'system', 'content': system_prompt},
             current_turn_message
        ]

        try:
            llm_response = safe_ollama_chat(prompt_messages)
            logging.debug(f"LLM Raw Response for Troubleshooter: {llm_response}")

            # --- FIX 4: Stricter Question Detection & Handling OOS-like responses ---
            is_question = False
            llm_lower = llm_response.lower().strip()
            is_likely_oos_response = "i specialize in providing support" in llm_lower or "related to these topics" in llm_lower

            if is_likely_oos_response:
                logging.warning(f"LLM generated an OOS-like response ('{llm_response}') despite troubleshooting context. Falling back.")
                return "Okay, I need a bit more information to understand the issue. Could you tell me the exact model of your device and the version of the software or OS you are using?", "ASK_CLARIFICATION"

            # Check for question mark AND common question keywords, excluding command keywords.
            if '?' in llm_response:
                 question_keywords = ['what', 'which', 'when', 'where', 'who', 'how', 'why', ' is ', ' are ', ' do ', ' does ', ' did ', ' can you ', ' could you ', ' please provide ', ' tell me ', ' confirm if ', ' have you ', ' may i ask ']
                 command_keywords = ['try ', 'reboot ', 'restart ', 'check ', 'verify ', 'update ', 'install ', 'clear ', 'disable ', 'enable ']
                 # Requires question keyword AND NOT a command keyword, and be reasonably long
                 if any(qkw in llm_lower for qkw in question_keywords) and \
                    not any(ckw in llm_lower for ckw in command_keywords) and \
                    len(llm_response.split()) > 3:
                      is_question = True

            if is_question:
                llm_question_text = re.sub(r'^(Okay|Alright|Sure|To clarify|I need to know|Could you tell me|Let me ask|To check one thing)(,|:)?\s*', '', llm_response, flags=re.IGNORECASE).strip()

                if not llm_question_text:
                     logging.warning("LLM response intended as question was empty after cleaning. Falling back to generic clarification.")
                     return "Okay, I need a bit more information. Could you describe exactly what you see on the screen or any error message that appears?", "ASK_CLARIFICATION"

                if not llm_question_text.endswith('?'): llm_question_text += '?'

                generic_phrases = ["tell me more", "provide more details", "what is the issue", "what seems to be the problem", "how can i help", "how can i assist"]
                if any(phrase in llm_question_text.lower() for phrase in generic_phrases):
                    logging.warning(f"LLM generated a generic question despite instructions: '{llm_question_text}'. Attempting fallback step.")
                    return "Okay, sometimes a simple restart can help after updates. Could you please try restarting your device first and let me know if that changes anything?", "PROVIDE_SOLUTION"

                logging.info(f"LLM asks specific question: '{llm_question_text}'")
                return llm_question_text, "ASK_CLARIFICATION"
            else:
                llm_step_text = re.sub(r'^(Okay|Alright|Sure|Try this|Next,|Let\'s try|Here\'s something to try|I suggest|Please try|Based on.*?,?|You could try)(,|:)?\s*', '', llm_response, flags=re.IGNORECASE).strip()
                llm_step_text = llm_step_text.split('\n')[0].strip()
                sentences = re.split(r'(?<=[.!?])\s+', llm_step_text)
                llm_step_text = sentences[0].strip() if sentences else llm_step_text

                if llm_step_text and not llm_step_text.endswith(('.', '!', '?', ':')): llm_step_text += '.'

                if not llm_step_text or len(llm_step_text.split()) < 3:
                    logging.warning(f"LLM response intended as solution was invalid/too short after cleaning: '{llm_response}'. Falling back to asking for specifics.")
                    return "I couldn't determine a specific step yet. Could you tell me the exact model of your device and the version of the operating system?", "ASK_CLARIFICATION"

                generic_phrases = ["what is the issue", "how can i help", "how can i assist"]
                if any(phrase in llm_step_text.lower() for phrase in generic_phrases) or llm_step_text.endswith('?'):
                     logging.warning(f"LLM generated a question/generic phrase ('{llm_step_text}') when a step was expected. Falling back.")
                     return "Okay, I need a bit more information. Could you describe exactly what you see on the screen or any error message that appears?", "ASK_CLARIFICATION"

                logging.info(f"LLM suggests step: '{llm_step_text}'")
                return llm_step_text, "PROVIDE_SOLUTION"

        except Exception as e:
            logging.error(f"LLM error during troubleshooting step/question generation: {e}.")
            return "I encountered an issue trying to determine the next step. To help me reassess, could you please describe what happens step-by-step when you experience the problem?", "ASK_CLARIFICATION"


class RoutingAgent:
    """Determines internal routing status based on the agent's last action type."""
    def route_ticket(self, issue_category: str, priority: str, action_type: str) -> str:
        """Maps the agent's action type to an internal status string."""
        status_map = {
            "ASK_CLARIFICATION": "Awaiting User Clarification",
            "GREETING": "Awaiting User Query",
            "AWAITING_FEEDBACK": "Awaiting User Feedback (Yes/No on Solution)",
            "FEEDBACK_POSITIVE": "Resolution Confirmed by User",
            "FEEDBACK_NEGATIVE": "Solution Failed - Awaiting Details",
            "PROVIDE_INFO": "Information Provided",
            "OUT_OF_SCOPE": "Closed - Query Out of Scope",
            "ERROR_LLM_CONNECTION": "Escalate - System Error (LLM)",
            "ERROR_INVALID_INPUT": "Review Needed - Invalid Input",
            "ERROR_UNEXPECTED": "Escalate - System Error (Unexpected)",
            "INTERNAL_ERROR_STATE": "Escalate - Internal Logic Error",
        }

        if action_type == "PROVIDE_SOLUTION":
             return status_map["AWAITING_FEEDBACK"]

        default_status = "Internal Review Needed"
        status = status_map.get(action_type, default_status)

        if status == default_status:
            logging.warning(f"RoutingAgent: Unhandled action_type '{action_type}' for status mapping. Defaulting to '{default_status}'.")

        logging.debug(f"Routing Status determined: '{status}' based on Action Type '{action_type}'")
        return status


class TimeEstimator:
    """Provides ETA based on historical data or static rules."""
    def __init__(self, historical_df: pd.DataFrame, is_time_data_valid: bool):
        self.historical_df = historical_df if isinstance(historical_df, pd.DataFrame) else pd.DataFrame()
        self.is_time_data_valid = is_time_data_valid
        self.avg_times = {}
        self.time_col = 'Resolution Time (hours)'
        self._calculate_average_times()

    def _calculate_average_times(self):
        if not self.is_time_data_valid or self.historical_df.empty:
            logging.warning("TimeEst: Skipping average time calculation as historical time data is invalid or unavailable.")
            self.avg_times = {}
            return

        required_cols_for_avg = ['Issue Category', 'Priority', self.time_col]
        if not all(c in self.historical_df.columns for c in required_cols_for_avg):
            logging.error(f"TimeEst: Logic Error! Time data flagged valid, but DataFrame missing required columns ({required_cols_for_avg}). Cannot calculate averages.")
            self.avg_times = {}
            return
        if not pd.api.types.is_numeric_dtype(self.historical_df[self.time_col]):
            logging.error(f"TimeEst: Logic Error! Time data flagged valid, but time column '{self.time_col}' is not numeric. Cannot calculate averages.")
            self.avg_times = {}
            return

        df_clean = self.historical_df.copy()
        df_clean['Priority'] = df_clean['Priority'].astype(str)
        df_clean['Issue Category'] = df_clean['Issue Category'].astype(str)

        if df_clean.empty:
            logging.warning("TimeEst: No valid rows remained after final checks for average time calculation.")
            self.avg_times = {}
            return

        try:
            means = df_clean.groupby(['Issue Category', 'Priority'], observed=False)[self.time_col].mean()

            valid_avg_count = 0
            self.avg_times = {}
            for (cat, prio), avg_time in means.items():
                 cat_str, prio_str = str(cat), str(prio) # Ensure keys are strings
                 if pd.notna(avg_time) and np.isfinite(avg_time) and avg_time >= 0:
                     if cat_str not in self.avg_times:
                         self.avg_times[cat_str] = {}
                     self.avg_times[cat_str][prio_str] = avg_time
                     valid_avg_count +=1
                 else:
                      logging.debug(f"TimeEst: Skipping invalid average time ({avg_time}) for {cat_str}/{prio_str}.")


            if valid_avg_count > 0:
                logging.info(f"TimeEst: Calculated {valid_avg_count} valid historical average resolution times across {len(self.avg_times)} categories.")
            else:
                 logging.warning("TimeEst: No valid historical average times could be calculated from the data.")

        except Exception as e:
            logging.error(f"TimeEst: Error during average time calculation: {e}", exc_info=True)
            self.avg_times = {}

    def estimate_resolution_time(self, issue_category: str, priority: str, action_type: str) -> str:
        """Generates an estimated time string based on historical data or static rules."""
        no_eta_states = ["GREETING", "PROVIDE_INFO", "AWAITING_FEEDBACK", "FEEDBACK_POSITIVE", "FEEDBACK_NEGATIVE", "OUT_OF_SCOPE"]
        if action_type in no_eta_states:
            return ""

        priority_lookup = str(priority or 'Medium').strip()
        category_lookup = str(issue_category or 'General Inquiry').strip()

        avg_hours = None
        eta_source = ""

        if self.is_time_data_valid and self.avg_times:
             avg_hours = self.avg_times.get(category_lookup, {}).get(priority_lookup)

        base_eta_text = ""

        if avg_hours is not None and avg_hours >= 0:
            eta_source = "hist. avg"
            if avg_hours < 1:
                 minutes = max(15, int(round(avg_hours * 60 / 15) * 15))
                 base_eta_text = f"around {minutes} minutes"
            elif avg_hours <= 8:
                 base_eta_text = f"around {int(round(avg_hours))} hour{'s' if round(avg_hours) > 1 else ''}"
            elif avg_hours <= 40:
                 days = max(1, int(round(avg_hours / 8)))
                 base_eta_text = f"around {days} business day{'s' if days > 1 else ''}"
            else:
                 weeks = max(1, int(round(avg_hours / 40)))
                 base_eta_text = f"around {weeks} week{'s' if weeks > 1 else ''}"
            logging.info(f"ETA generated from historical data ({category_lookup}/{priority_lookup}): {avg_hours:.1f}h -> {base_eta_text}")
        else:
            eta_source = "std."
            logging.info(f"No valid historical average found for '{category_lookup}'/'{priority_lookup}'. Using static priority-based estimate.")
            prio_low = priority_lookup.lower()
            if prio_low == 'critical': base_eta_text = "within 2-4 hours"
            elif prio_low == 'high': base_eta_text = "within 1 business day"
            elif prio_low == 'medium': base_eta_text = "within 1-3 business days"
            elif prio_low == 'low': base_eta_text = "within 3-5 business days"
            else:
                 base_eta_text = "within 3-5 business days"
                 logging.warning(f"TimeEst: Unknown priority '{priority_lookup}' encountered for static estimate. Defaulting.")

        action_qualifier = ""
        if action_type == "ASK_CLARIFICATION":
            action_qualifier = "once we have the necessary details"
        elif action_type == "PROVIDE_SOLUTION":
             return ""

        if base_eta_text and action_qualifier:
             return f"Estimated resolution time {action_qualifier}: {base_eta_text} ({eta_source})."
        elif base_eta_text:
             return f"Estimated resolution time: {base_eta_text} ({eta_source})."
        else:
             return ""


class CategoryPredictor:
    """Predicts category based on query, using LLM and matching against available categories."""
    def predict_category(self, query: str, available_categories: list) -> str:
        """Predicts the most likely support category for a given query."""
        default_category="General Inquiry"
        display_categories = sorted(list(set([default_category] + (available_categories or []))))

        if len(display_categories) <= 1:
             logging.warning("CategoryPredictor: No specific categories loaded from KB. Defaulting all queries to 'General Inquiry'.")
             return default_category

        categories_str = "\n - ".join(display_categories)
        prompt = (f"Analyze the user's support query below. Determine the single BEST matching support category from the following list. "
                  f"Focus on the main technical area or type of problem described.\n"
                  f"If the query is very general, a simple greeting, out of scope, or doesn't clearly fit any specific category, choose '{default_category}'.\n\n"
                  f"Available Categories:\n - {categories_str}\n\n"
                  f"User Query: \"{query}\"\n\n"
                  f"Respond ONLY with the exact category name from the list.")
        try:
            messages = [{'role': 'user', 'content': prompt}]
            predicted_raw = safe_ollama_chat(messages).strip().rstrip('.')

            cat_map = {cat.lower(): cat for cat in display_categories}
            predicted_clean = predicted_raw.lower()

            if predicted_clean in cat_map:
                best_match = cat_map[predicted_clean]
                logging.info(f"CatPredict: LLM chose '{predicted_raw}' -> Mapped to '{best_match}' (Exact Match)")
                return best_match
            else:
                found_match = None
                for cat_lower, cat_original in cat_map.items():
                    if re.search(r'\b' + re.escape(cat_lower) + r'\b', predicted_clean):
                         found_match = cat_original
                         logging.warning(f"CatPredict: LLM response '{predicted_raw}' not exact match, but contains '{cat_original}'. Using as fallback.")
                         break

                if found_match:
                    return found_match

                logging.warning(f"CatPredict: LLM response '{predicted_raw}' did not match or contain any known category. Defaulting to '{default_category}'.")
                return default_category

        except Exception as e:
            logging.error(f"CatPredict Error: {e}. Defaulting to '{default_category}'.")
            return default_category


class SentimentAnalyzer:
    """Analyzes user sentiment from query using LLM."""
    def analyze_sentiment(self, query: str) -> str:
        """Predicts user sentiment from the query text."""
        default_sentiment = "Neutral"
        if not query: return default_sentiment

        valid_sentiments = ["Frustrated", "Confused", "Annoyed", "Anxious", "Urgent", "Neutral", "Positive"]
        sentiments_str = ", ".join(valid_sentiments)

        prompt = (f"Analyze the sentiment expressed in the user's support query. Consider the tone, keywords (like 'urgent', 'hate', 'confused', 'thanks', 'great'), and punctuation (like '!!!'). "
                  f"If the sentiment is unclear, mixed, purely factual, or just a greeting, choose '{default_sentiment}'. "
                  f"Choose exactly ONE sentiment from the following list: {sentiments_str}.\n\n"
                  f"User Query: \"{query}\"\n\n"
                  f"Sentiment:")
        try:
            messages = [{'role': 'user', 'content': prompt}]
            response_raw = safe_ollama_chat(messages).strip().rstrip('.')
            response_clean = response_raw.lower()
            match = None

            for s in valid_sentiments:
                if response_clean == s.lower():
                    match = s
                    break

            if not match:
                 for s in valid_sentiments:
                     if re.search(r'\b' + re.escape(s.lower()) + r'\b', response_clean):
                         logging.warning(f"Sentiment: LLM response '{response_raw}' not exact, using contained sentiment '{s}'.")
                         match = s
                         break

            if match:
                logging.info(f"Sentiment Analyzed: {match}")
                return match
            else:
                logging.warning(f"Sentiment Analyzer couldn't map LLM response '{response_raw}' to valid sentiments. Defaulting to {default_sentiment}.")
                return default_sentiment
        except Exception as e:
            logging.error(f"Sentiment Analyzer Error: {e}. Defaulting to {default_sentiment}.")
            return default_sentiment


class PriorityPredictor:
    """Predicts priority based on query content, keywords, and sentiment."""
    def predict_priority(self, query: str, sentiment: str) -> str:
        """Assigns a priority level (Low, Medium, High, Critical) to the query."""
        default_priority = "Medium"
        valid_priorities = ["Low", "Medium", "High", "Critical"]
        priorities_str = ", ".join(valid_priorities)

        sentiment_norm = (sentiment or default_priority).lower()
        query_lower = (query or "").lower()
        if not query_lower: return "Low"

        # --- Keyword-based Overrides (for immediate critical/high flags) ---
        critical_kws = [
            'critical', 'urgent', 'blocker', 'down', 'outage', 'cannot process payment', 'system offline',
             'security breach', 'vulnerability', 'data loss', 'halted', 'emergency', 'cannot operate', 'production issue'
        ]
        high_kws = [
            'cannot work', 'major issue', 'payment failure', 'significant disruption', 'severe bug',
            'multiple users affected', 'degraded performance', 'stuck process', 'repeated failures',
            'failing consistently', 'unable to complete'
        ]

        matched_critical_kw = next((kw for kw in critical_kws if kw in query_lower), None)
        if matched_critical_kw:
            logging.info(f"Priority KW Override: Critical due to keyword '{matched_critical_kw}'")
            return "Critical"

        matched_high_kw = next((kw for kw in high_kws if kw in query_lower), None)
        if matched_high_kw and sentiment_norm in ['frustrated', 'annoyed', 'anxious', 'urgent', 'critical']:
             logging.info(f"Priority KW Override: High due to keyword '{matched_high_kw}' and sentiment '{sentiment}'")
             return "High"

        # --- LLM-based Prediction (if no keyword override) ---
        prompt = (f"Analyze the user query and its expressed sentiment ('{sentiment}') to determine the support priority level. Choose ONE from: {priorities_str}.\n"
                  f"Consider these guidelines:\n"
                  f"- Critical: Complete service outage, security exploit, data corruption/loss, user cannot operate system AT ALL, payment processing blocked entirely.\n"
                  f"- High: Major function is broken or unusable, significantly impacting workflow for one or multiple users, persistent errors preventing tasks.\n"
                  f"- Medium: Standard functional issue, workaround might exist but inconvenient, user experiencing frustration/confusion, performance issues.\n"
                  f"- Low: 'How-to' question, minor cosmetic bug, request for information, feature request, user has neutral/positive sentiment.\n\n"
                  f"User Query: \"{query}\"\n\n"
                  f"Priority:")
        try:
            messages = [{'role': 'user', 'content': prompt}]
            response_raw = safe_ollama_chat(messages).strip().rstrip('.')
            response_clean = response_raw.lower()
            match = None

            for p in valid_priorities:
                if response_clean == p.lower():
                    match = p
                    break

            if not match:
                 for p in valid_priorities:
                     if re.search(r'\b' + re.escape(p.lower()) + r'\b', response_clean):
                         logging.warning(f"Priority: LLM response '{response_raw}' not exact, using contained priority '{p}'.")
                         match = p
                         break

            if match:
                 is_potentially_trivial = any(kw in query_lower for kw in ['how do i', 'where is', 'minor', 'suggestion', 'typo'])
                 if match in ["Critical", "High"] and (is_potentially_trivial or sentiment_norm == 'positive'):
                      if not matched_critical_kw and not matched_high_kw: # Only adjust if not triggered by strong keywords
                          new_prio = "High" if match == "Critical" else "Medium"
                          logging.warning(f"Downgrading LLM Prio '{match}' to '{new_prio}' due to trivial query indicators or positive sentiment without strong keywords.")
                          return new_prio

                 logging.info(f"Priority (LLM Prediction): {match}")
                 return match
            else:
                logging.warning(f"Priority LLM response unclear or unmappable: '{response_raw}'. Using sentiment-based fallback.")
                if sentiment_norm == 'urgent': return "Critical"
                if sentiment_norm == 'anxious': return "High"
                if sentiment_norm in ['frustrated', 'annoyed', 'confused']: return "Medium"
                return "Low"

        except Exception as e:
             logging.error(f"Priority Predictor Error: {e}. Using sentiment-based fallback.")
             if sentiment_norm == 'urgent': return "Critical"
             if sentiment_norm == 'anxious': return "High"
             if sentiment_norm in ['frustrated', 'annoyed', 'confused']: return "Medium"
             return "Low"


class ResponseAdjuster:
    """Adds an empathetic prefix to the response based on sentiment."""
    def adjust_response(self, response_core: str, sentiment: str) -> str:
        """Prepends an empathetic phrase if sentiment is negative/urgent."""
        sentiment_norm = (sentiment or "neutral").lower()
        prefix_map = {
            'frustrated': "I understand this is frustrating, let's see how we can sort this out.\n\n",
            'annoyed': "I can see why that would be annoying. Let's work through it together.\n\n",
            'anxious': "I understand this might be concerning. I'll do my best to help you resolve it.\n\n",
            'urgent': "Acknowledging the urgency here. Let's investigate this promptly.\n\n",
            'confused': "It can definitely be confusing sometimes. Let's clarify this step by step.\n\n",
        }
        # Get the prefix for the detected sentiment, default to empty string if neutral/positive/unmapped
        prefix = prefix_map.get(sentiment_norm, "")
        # Combine prefix and core response, ensuring core response doesn't have leading/trailing space
        return prefix + response_core.strip()


class CMARController:
    def __init__(self, historical_df: pd.DataFrame, knowledge_base: dict, df_time_valid: bool):
        """Initializes the CMAR controller with data and agents."""
        logging.info(f"Initializing CMARController (LLM: {TEXT_MODEL})...")
        self.historical_df = historical_df if isinstance(historical_df, pd.DataFrame) else pd.DataFrame()
        self.knowledge_base = knowledge_base if isinstance(knowledge_base, dict) else {}
        self.df_time_valid = df_time_valid
        self.time_col = 'Resolution Time (hours)'

        self.category_predictor = CategoryPredictor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.priority_predictor = PriorityPredictor()
        self.initial_troubleshooter = InitialTroubleshooter(self.knowledge_base)
        self.response_adjuster = ResponseAdjuster()
        self.time_estimator = TimeEstimator(self.historical_df, self.df_time_valid)
        self.summary_agent = SummaryAgent()
        self.routing_agent = RoutingAgent()
        self.action_extractor = ActionExtractor()

        self.memory = {}
        logging.info(f"CMARController initialized successfully. KB Categories: {len(self.knowledge_base)}. Historical time data valid: {self.df_time_valid}.")

    def is_greeting_only(self, query: str) -> bool:
        """Checks if the query is likely just a greeting without a problem statement."""
        query_lower = query.lower().strip().rstrip('.!?')
        if not query_lower: return False

        greetings = [
            'hello', 'hi', 'hey', 'heya', 'yo', 'greetings', 'good morning', 'good afternoon',
            'good evening', 'morning', 'afternoon', 'evening', 'namaste', 'salaam', 'bonjour',
            'hola', 'hi there', 'hello there', 'sup', 'howdy', 'hiya'
        ]
        if query_lower in greetings: return True

        words = query_lower.split()
        if len(words) <= 3 and any(greet in words for greet in greetings):
             # Check if it *also* contains problem keywords - if so, it's not *just* a greeting
             problem_keywords = ['problem', 'issue', 'error', 'help', 'support', 'broken', 'fail', 'login', 'sync', 'install', 'update', 'payment', 'not working']
             if not any(kw in query_lower for kw in problem_keywords):
                  logging.debug(f"Query identified as greeting only: '{query}'")
                  return True
        return False

    def generate_greeting_response(self, query: str) -> str:
        """Generates a friendly greeting response using LLM."""
        prompt = f"The user initiated the chat with a greeting: \"{query}\". Respond with a brief, friendly, welcoming greeting (1-2 sentences) and ask how you can help them with our software/service today. Sound professional but approachable."
        try:
            messages = [{'role': 'user', 'content': prompt}]
            # Use the main text model for consistency
            response = safe_ollama_chat(messages, model=TEXT_MODEL)
            return response.strip() or "Hello! How can I assist you with our services today?"
        except Exception as e:
            logging.error(f"Greeting response generation error: {e}")
            return "Hello! How can I help you today?"

    def update_knowledge_base(self, category: str, solution: str):
        """
        Adds a new successfully verified solution to the in-memory KB.
        (Note: This update is temporary and lost when the server restarts unless persisted).
        """
        category = (category or "").strip()
        solution = (solution or "").strip()

        if not category or category == "General Inquiry" or not solution:
            logging.debug(f"KB Update skipped: Invalid category ('{category}') or empty solution.")
            return

        if category in self.knowledge_base:
            existing_solutions_lower = {s.lower() for s in self.knowledge_base[category]}
            if solution.lower() not in existing_solutions_lower:
                self.knowledge_base[category].append(solution)
                logging.info(f"KB Updated: Added new solution to existing category '{category}'.")
            else:
                 logging.debug(f"KB Update: Solution likely already exists (case-insensitive) in category '{category}'. No changes made.")
        else:
            self.knowledge_base[category] = [solution]
            logging.info(f"KB Updated: Added new category '{category}' with its first solution.")

        # IMPORTANT: Update the troubleshooter instance with the modified KB reference
        self.initial_troubleshooter.knowledge_base = self.knowledge_base
        logging.debug(f"InitialTroubleshooter KB reference updated. Total categories now: {len(self.knowledge_base)}")


    # --- FIX 3a: Modify is_query_support_related prompt and add history param ---
    def is_query_support_related(self, query: str, conversation_history: list = None) -> bool:
        """Uses LLM to check if the query is relevant to the expected support domain, considering history."""
        history_context = "This is the first message."
        if conversation_history:
             history_context = f"There is existing conversation history (last few turns shown to you before this check). The user's previous message might be a follow-up."

        prompt = (f"""Analyze the user's latest query in the context of a support chat.
Is the query related to seeking help, reporting a problem, asking for information about a specific software/service/account/billing, OR is it a direct follow-up or response to a previous question within this support conversation?

Context: {history_context}

Examples of relevant queries:
- "My app keeps crashing."
- "How do I reset my password?"
- "I think I was billed incorrectly."
- "The installation failed with error code 123."
- "What are the system requirements?"
- "Yes, I tried that." (Follow-up)
- "It still doesn't work." (Follow-up)
- "What do you mean by 'force restart'?" (Follow-up/Clarification)

Examples of irrelevant queries (if they are the *start* of a conversation or unrelated to previous turns):
- "What's the weather today?"
- "Can you write a poem?"
- "Hello" (greeting only, handled separately)
- "Tell me a joke."
- "What is your opinion on AI?"

User's Latest Query: "{query}"

Answer ONLY 'Yes' (relevant/follow-up) or 'No' (irrelevant/off-topic).
""")
        try:
            context_for_check = conversation_history[-2:] if conversation_history and len(conversation_history) >=2 else []
            messages = context_for_check + [{'role': 'user', 'content': prompt}]
            response = safe_ollama_chat(messages)
            response_clean = response.strip().strip('.!?').lower()

            if response_clean == 'yes':
                logging.info(f"Domain Check: Query seems support-related/follow-up. Query='{query[:100]}...'")
                return True
            elif response_clean == 'no':
                 logging.warning(f"Domain Check: Query determined NOT support-related (LLM='{response}'). Query='{query[:100]}...'")
                 return False
            else:
                 logging.warning(f"Domain Check: LLM response ambiguous ('{response}'). Assuming support-related as a cautious fallback.")
                 return True
        except Exception as e:
            logging.error(f"Domain check LLM error: {e}. Assuming query IS support-related as fallback.")
            return True

    # --- Main Query Processing Method ---
    def process_query(self, convo_id: str, query: str) -> dict:
        """Processes a user query, manages state, calls agents, and returns agent response."""
        start_time = datetime.now()
        logging.info(f"--- Query START: Convo ID '{convo_id}' Query: '{query[:100]}...' ---")

        if not isinstance(query, str) or not query.strip():
            logging.error(f"Invalid query received for Convo ID '{convo_id}': Query is not a non-empty string.")
            raise ValueError("Query must be a non-empty string.")

        if convo_id not in self.memory:
            logging.info(f"New conversation started: ID '{convo_id}'.")
            self.memory[convo_id] = []

        if self.is_greeting_only(query):
            logging.info(f"Handling 'Greeting Only' for Convo ID '{convo_id}'.")
            greeting_response = self.generate_greeting_response(query)
            customer_turn = {'role': 'user', 'content': query, 'timestamp': datetime.now()}
            agent_turn = {'role': 'assistant', 'content': greeting_response, 'timestamp': datetime.now(), 'action_type': 'GREETING'}
            self.memory[convo_id].extend([customer_turn, agent_turn])
            proc_time = (datetime.now() - start_time).total_seconds()
            internal_log_details = [
                f"Processing Time: {proc_time:.2f}s",
                "Action: Handled Greeting Only.",
                f"Response: {greeting_response}"
            ]
            return {
                "reply": greeting_response,
                "convo_id": convo_id,
                "internal_log": "\n".join(internal_log_details),
                "ask_feedback": False
            }

        # --- 2. Check if Query is Support-Related (Domain Check) ---
        # --- FIX 3b: Improved Logic to Skip Domain Check ---
        perform_domain_check = True
        last_agent_action = None
        if self.memory[convo_id]:
            try:
                last_turn = self.memory[convo_id][-1]
                if last_turn.get('role') == 'assistant':
                     last_agent_action = last_turn.get('action_type')
                     # Skip domain check if the agent just asked a question, is awaiting feedback, or responded to feedback.
                     if last_agent_action in ['ASK_CLARIFICATION', 'AWAITING_FEEDBACK', 'FEEDBACK_POSITIVE', 'FEEDBACK_NEGATIVE']:
                         logging.info(f"Skipping domain check for follow-up message (Agent action: {last_agent_action}). Convo '{convo_id}'.")
                         perform_domain_check = False
                     elif last_agent_action == 'OUT_OF_SCOPE':
                          logging.info(f"Skipping domain check following agent's OUT_OF_SCOPE response. Convo '{convo_id}'.")
                          perform_domain_check = False

            except IndexError:
                pass
            except Exception as e:
                logging.error(f"Error checking memory for domain skip condition: {e}. Proceeding with check.")

        if perform_domain_check:
            try:
                # Pass conversation history to the domain check function
                history_for_domain_check = self.memory.get(convo_id, [])
                if not self.is_query_support_related(query, history_for_domain_check):
                    log_msg = "Query determined to be out of scope / not support-related."
                    response = "I specialize in providing support for our software and services (e.g., technical issues, account management, billing, how-to questions). Could you please ask a question related to these topics?"
                    logging.warning(f"{log_msg} Convo ID: {convo_id}, Query: '{query[:100]}...'")
                    customer_turn = {'role': 'user', 'content': query, 'timestamp': datetime.now()}
                    agent_turn = {'role': 'assistant', 'content': response, 'timestamp': datetime.now(), 'action_type': 'OUT_OF_SCOPE'}
                    self.memory[convo_id].extend([customer_turn, agent_turn])
                    proc_time = (datetime.now() - start_time).total_seconds()
                    internal_log_details = [
                        f"Processing Time: {proc_time:.2f}s",
                        log_msg,
                        f"Agent Response: {response}"
                    ]
                    return {
                        "reply": response,
                        "convo_id": convo_id,
                        "internal_log": "\n".join(internal_log_details),
                        "ask_feedback": False
                    }
            # Handle potential LLM errors during the domain check itself
            except (ConnectionError, RuntimeError) as llm_error:
                 logging.error(f"LLM Error during domain check for Convo ID '{convo_id}': {llm_error}. Proceeding with query processing as fallback.")
            except Exception as e:
                 # Catch other unexpected errors during domain check
                 logging.error(f"Unexpected Error during domain check for Convo ID '{convo_id}': {e}. Proceeding with query processing as fallback.")


        history_for_prompt = [
            {'role': m['role'], 'content': m['content']}
            for m in self.memory[convo_id][-(HISTORY_TURNS * 2):]
        ]
        customer_turn_data = {'role': 'user', 'content': query, 'timestamp': datetime.now()}
        internal_log_details = [
            f"--- Turn Analysis Start: {convo_id} ---",
            f"User Query: {query}",
            f"History Turns Provided to LLM: {len(history_for_prompt)//2} pairs ({len(history_for_prompt)} messages)"
        ]

        try:
            available_categories = list(self.knowledge_base.keys())
            predicted_category = self.category_predictor.predict_category(query, available_categories)
            sentiment = self.sentiment_analyzer.analyze_sentiment(query)
            predicted_priority = self.priority_predictor.predict_priority(query, sentiment)
            summary = self.summary_agent.generate_summary(query)

            internal_log_details.extend([
                f"Predicted Category: '{predicted_category}'",
                f"Predicted Sentiment: {sentiment}",
                f"Predicted Priority: {predicted_priority}",
                f"Internal Summary: {summary}"
            ])

            resolution_text, resolution_action_type = self.initial_troubleshooter.suggest_initial_steps(
                predicted_category, query, history_for_prompt
            )
            internal_log_details.append(f"Troubleshooter Raw Action Type: {resolution_action_type}")
            internal_log_details.append(f"Troubleshooter Raw Response Text:\n{resolution_text}")

            routing_status = self.routing_agent.route_ticket(predicted_category, predicted_priority, resolution_action_type)
            estimated_time = self.time_estimator.estimate_resolution_time(predicted_category, predicted_priority, resolution_action_type)

            internal_log_details.extend([
                f"Initial Routing Status: {routing_status}",
                f"Generated ETA (if applicable): {estimated_time or 'N/A'}"
            ])

            response_prefix = self.response_adjuster.adjust_response("", sentiment) # Get empathetic prefix if needed
            display_focus = f"the '{predicted_category}' issue" if predicted_category and predicted_category != "General Inquiry" else "your query"

            # Start with potential prefix, core response text will be added based on action type
            response_body = response_prefix

            ask_feedback_flag = False
            final_action_type = resolution_action_type

            if resolution_action_type == "ASK_CLARIFICATION":
                if resolution_text:
                     # Prepend intro if no empathetic prefix exists
                     if not response_prefix: response_body += "Okay, to help me understand better:\n\n"
                     response_body += f"**{resolution_text}**\n\n"
                     if estimated_time: response_body += f"{estimated_time}"
                else:
                     logging.error(f"ASK_CLARIFICATION action type returned empty text for Convo ID '{convo_id}'. Sending generic clarification.")
                     # Ensure prefix is not duplicated if it exists
                     response_body = response_prefix + "**Apologies, I need a bit more information. Could you please describe the issue in more detail?**\n\n"

            elif resolution_action_type == "PROVIDE_SOLUTION":
                if resolution_text:
                    # Prepend intro if no empathetic prefix exists
                    if not response_prefix: response_body += "Okay, let's try this step:\n\n"
                    response_body += f"**Here's the next step I recommend trying:**\n\n- {resolution_text}\n\n"
                    response_body += "**Please let me know if that step worked for you. You can simply reply with 'yes' or 'no'.**"
                    final_action_type = "AWAITING_FEEDBACK"
                    ask_feedback_flag = True
                    routing_status = self.routing_agent.route_ticket(predicted_category, predicted_priority, final_action_type)
                else:
                    logging.error(f"PROVIDE_SOLUTION action type returned empty text for Convo ID '{convo_id}'. Switching to clarification.")
                    response_body = response_prefix + "**I seem to be having trouble finding the right step. Could you please describe exactly what you see on the screen when the issue occurs?**\n\n"
                    final_action_type = "ASK_CLARIFICATION"
                    ask_feedback_flag = False
                    routing_status = self.routing_agent.route_ticket(predicted_category, predicted_priority, final_action_type)

            elif resolution_action_type == "PROVIDE_INFO":
                if resolution_text:
                     # Prepend intro if no empathetic prefix exists
                     if not response_prefix: response_body += "Okay, regarding your question:\n\n"
                     response_body += f"**Regarding your question:**\n\n{resolution_text}\n\n"
                     response_body += "Does that answer your question, or is there anything else I can help you with related to our services?"
                else:
                     logging.error(f"PROVIDE_INFO action type returned empty text for Convo ID '{convo_id}'. Sending generic response.")
                     response_body = response_prefix + "**I couldn't retrieve that specific information at the moment. Could you perhaps ask in a different way or clarify what you need?**\n\n"
                     final_action_type = "ASK_CLARIFICATION"
                     routing_status = self.routing_agent.route_ticket(predicted_category, predicted_priority, final_action_type)

            else:
                 logging.error(f"Unexpected resolution_action_type '{resolution_action_type}' received in CMARController for Convo ID '{convo_id}'. Providing generic response.")
                 # Construct fallback response, including prefix if present
                 response_body = response_prefix + f"I've noted your request regarding '{display_focus}'. "
                 estimated_time_fallback = self.time_estimator.estimate_resolution_time(predicted_category, predicted_priority, "ASK_CLARIFICATION")
                 if estimated_time_fallback:
                     response_body += f"We'll investigate this further. {estimated_time_fallback}."
                 else:
                     response_body += "We will look into this and follow up as soon as possible."
                 final_action_type = "INTERNAL_ERROR_STATE"
                 routing_status = self.routing_agent.route_ticket(predicted_category, predicted_priority, final_action_type)


            final_response = response_body.strip()
            agent_actions_log = self.action_extractor.extract_actions(query, final_action_type)
            internal_log_details.append(f"Final Action Type: {final_action_type}")
            internal_log_details.append(f"Final Routing Status: {routing_status}")
            internal_log_details.append(f"Logged Agent Action Description: {agent_actions_log}")

            agent_turn_data = {
                'role': 'assistant',
                'content': final_response,
                'timestamp': datetime.now(),
                'predicted_category': predicted_category,
                'sentiment': sentiment,
                'priority': predicted_priority,
                'internal_summary': summary,
                'resolution_text_provided': resolution_text,
                'action_type': final_action_type,
                'internal_routing_status': routing_status,
                'internal_eta_generated': estimated_time,
                'internal_agent_actions_log': agent_actions_log,
                'ask_feedback': ask_feedback_flag
            }
            self.memory[convo_id].extend([customer_turn_data, agent_turn_data])

            proc_time = (datetime.now() - start_time).total_seconds()
            internal_log_details.insert(1, f"Processing Time: {proc_time:.2f}s")
            internal_log_str = "\n".join(internal_log_details) + f"\n--- Agent Response Sent ---\n{final_response}\n--- Turn Analysis End: {convo_id} ---"

            logging.info(f"--- Query END: Convo '{convo_id}'. Final Action: {final_action_type}. Proc Time: {proc_time:.2f}s ---")

            return {
                "reply": final_response,
                "convo_id": convo_id,
                "internal_log": internal_log_str,
                "ask_feedback": ask_feedback_flag
             }

        except (ConnectionError, RuntimeError) as agent_error:
            error_message = f"LLM/Connection Error processing query for Convo ID '{convo_id}': {agent_error}"
            logging.error(error_message, exc_info=False)
            error_response = "I'm currently experiencing difficulties connecting to my core systems. Please try your request again in a few moments."
            customer_turn_data = {'role': 'user', 'content': query, 'timestamp': datetime.now()}
            agent_turn_data = {'role': 'assistant', 'content': error_response, 'timestamp': datetime.now(), 'action_type': 'ERROR_LLM_CONNECTION'}
            if convo_id in self.memory: self.memory[convo_id].extend([customer_turn_data, agent_turn_data])
            internal_log_details.append(f"!! PIPELINE FAILED (LLM/Connection Error): {agent_error}")
            raise agent_error

        except ValueError as val_error:
             error_message = f"Value Error during query processing for Convo ID '{convo_id}': {val_error}"
             logging.warning(error_message, exc_info=True)
             error_response = f"There was an issue processing your request due to invalid data: {val_error}. Please check your input."
             customer_turn_data = {'role': 'user', 'content': query, 'timestamp': datetime.now()}
             agent_turn_data = {'role': 'assistant', 'content': error_response, 'timestamp': datetime.now(), 'action_type': 'ERROR_INVALID_INPUT'}
             if convo_id in self.memory: self.memory[convo_id].extend([customer_turn_data, agent_turn_data])
             internal_log_details.append(f"!! PIPELINE FAILED (Value Error): {val_error}")
             raise val_error

        except Exception as e:
            error_message = f"CRITICAL UNEXPECTED ERROR during query processing for Convo ID '{convo_id}': {e}"
            logging.critical(error_message, exc_info=True)
            error_response = "I encountered an unexpected internal error while processing your request. Please try again later or contact support through another channel if the problem persists."
            customer_turn_data = {'role': 'user', 'content': query, 'timestamp': datetime.now()}
            agent_turn_data = {'role': 'assistant', 'content': error_response, 'timestamp': datetime.now(), 'action_type': 'ERROR_UNEXPECTED'}
            if convo_id in self.memory: self.memory[convo_id].extend([customer_turn_data, agent_turn_data])
            internal_log_details.append(f"!! PIPELINE FAILED (Unexpected Error): {e}")
            raise Exception(f"Unexpected internal error during query processing: {e}")


    def process_feedback(self, convo_id: str, success: bool) -> dict:
        """Processes user's yes/no feedback on the last proposed solution."""
        start_time = datetime.now()
        feedback_type = 'Yes (Success)' if success else 'No (Failure)'
        logging.info(f"--- Feedback START: Convo ID '{convo_id}', Feedback: {feedback_type} ---")

        if convo_id not in self.memory or not self.memory.get(convo_id):
            logging.warning(f"Feedback error for Convo ID '{convo_id}': No conversation history found.")
            proc_time = (datetime.now() - start_time).total_seconds()
            return {
                "reply": "I seem to have lost the context of our conversation. Could you please state your issue again?",
                "reward": 0,
                "internal_log": f"Feedback failed: No history found for convo '{convo_id}'. Proc Time: {proc_time:.2f}s"
            }

        last_agent_turn_index = -1
        original_solution_text = None
        original_category = None

        try:
            for i in range(len(self.memory[convo_id]) - 1, -1, -1):
                turn = self.memory[convo_id][i]
                if turn.get('role') == 'assistant' and \
                   turn.get('action_type') == 'AWAITING_FEEDBACK' and \
                   'feedback_received' not in turn:
                        last_agent_turn_index = i
                        original_solution_text = turn.get('resolution_text_provided')
                        original_category = turn.get('predicted_category')
                        logging.debug(f"Found agent turn at index {i} awaiting feedback for convo '{convo_id}'. Solution: '{original_solution_text}'")
                        break
                elif turn.get('role') == 'assistant' and 'feedback_received' in turn:
                     logging.debug(f"Turn {i} for convo '{convo_id}' already received feedback ({'Success' if turn['feedback_received'] else 'Failure'}). Continuing search.")

        except Exception as e:
            logging.error(f"Memory access error during feedback processing for '{convo_id}': {e}", exc_info=True)
            proc_time = (datetime.now() - start_time).total_seconds()
            return {
                "reply": "An error occurred while trying to process your feedback due to a memory issue. Please try again.",
                "reward": 0,
                "internal_log": f"Feedback failed: Memory access error. Proc Time: {proc_time:.2f}s"
            }

        if last_agent_turn_index == -1:
            logging.warning(f"No agent turn found actively 'AWAITING_FEEDBACK' (or feedback already processed) for convo '{convo_id}'. Ignoring current feedback.")
            proc_time = (datetime.now() - start_time).total_seconds()
            return {
                "reply": "Thanks for the update! If you have another question or issue, feel free to ask.",
                "reward": 0,
                "internal_log": f"Feedback ignored: No turn awaiting feedback found or feedback already processed. Proc Time: {proc_time:.2f}s"
            }

        try:
            self.memory[convo_id][last_agent_turn_index]['feedback_received'] = success
            logging.debug(f"Marked feedback_received={success} for turn {last_agent_turn_index} in convo '{convo_id}'.")
        except Exception as e:
             logging.error(f"Error marking feedback received in memory for turn {last_agent_turn_index}, convo '{convo_id}': {e}")

        response_payload = {"reward": 0}
        action_type_log = "FEEDBACK_POSITIVE" if success else "FEEDBACK_NEGATIVE"
        internal_log_details = [
            f"Feedback: '{feedback_type}' processed for agent turn at index {last_agent_turn_index}.",
            f"Original Solution Attempted: '{original_solution_text or 'N/A'}'",
            f"Original Predicted Category: '{original_category or 'N/A'}'"
        ]

        if success:
            response_payload["reward"] = 10
            if original_category and original_solution_text:
                try:
                    self.update_knowledge_base(original_category, original_solution_text)
                    internal_log_details.append(f"KB update successful for Category='{original_category}'.")
                except Exception as kb_update_err:
                     logging.error(f"Error updating KB during positive feedback for convo '{convo_id}': {kb_update_err}")
                     internal_log_details.append(f"KB update FAILED: {kb_update_err}")
            else:
                logging.warning(f"Feedback 'Yes' for convo '{convo_id}', but missing original data. Cannot update KB.")
                internal_log_details.append("KB Update skipped (missing original data).")

            response_payload["reply"] = (f"Excellent! I'm glad to hear that resolved the issue for you. 😊\n\n"
                                         f"Is there anything else I can help you with today regarding our services?")
            internal_log_details.append("Replied with success confirmation.")

        else:
            response_payload["reward"] = 0
            response_payload["reply"] = ("Okay, thanks for letting me know that didn't work. Sorry the issue isn't resolved yet.\n\n"
                                         "To help me find a different solution or understand what went wrong, could you please tell me:\n"
                                         "- What exactly happened when you tried the last step?\n"
                                         "- Did you see any specific error messages or notice anything unusual?")
            internal_log_details.append("Replied acknowledging failure and prompting for details.")

        feedback_user_turn = {
            'role': 'user',
            'content': f"Feedback on previous step: {'Yes, it worked.' if success else 'No, it did not work.'}",
            'timestamp': datetime.now(),
            'is_feedback': True
        }
        feedback_agent_turn = {
            'role': 'assistant',
            'content': response_payload["reply"],
            'timestamp': datetime.now(),
            'action_type': action_type_log
        }
        self.memory[convo_id].extend([feedback_user_turn, feedback_agent_turn])
        logging.debug(f"Added feedback user turn and agent response turn to memory for convo '{convo_id}'.")

        proc_time = (datetime.now() - start_time).total_seconds()
        internal_log_details.insert(0, f"Processing Time: {proc_time:.2f}s")
        response_payload["internal_log"] = "\n".join(internal_log_details)
        logging.info(f"--- Feedback END: Convo '{convo_id}'. Responded to feedback ({feedback_type}). Reward: {response_payload['reward']}. Proc Time: {proc_time:.2f}s ---")

        return response_payload


# --- Main Execution Guard (for standalone testing if needed) ---
if __name__ == "__main__":
    logging.info("--- backend_logic.py execution started directly for testing ---")
    try:
        logging.info(f"Loading KB from: {CSV_FILE_PATH}")
        historical_df, knowledge_base, time_valid = load_knowledge_base(CSV_FILE_PATH)

        if historical_df is None or knowledge_base is None:
             logging.critical("Initialization failed: Could not load data properly. Exiting test.")
             sys.exit(1)
        logging.info(f"KB Loaded ({len(knowledge_base)} categories). Time Data Valid: {time_valid}")

        controller = CMARController(historical_df, knowledge_base, time_valid)
        logging.info("CMARController initialized for testing.")

        convo_id = f"TEST_CONVO_{datetime.now().strftime('%H%M%S')}"
        print(f"\n--- Starting Test Conversation: {convo_id} ---")

        print("\n--- Test Query 1: Initial Problem ---")
        query1 = "Hey so my iphone has stoppped working suddenly after an update"
        print(f"User: {query1}")
        response1 = controller.process_query(convo_id, query1)
        print(f"Agent:\n{response1.get('reply')}")
        print(f"Ask Feedback: {response1.get('ask_feedback')}")

        if response1.get('ask_feedback'):
            print("\n--- Test Feedback (Scenario: No) ---")
            feedback_no_response = controller.process_feedback(convo_id, success=False)
            print(f"User: No")
            print(f"Agent:\n{feedback_no_response.get('reply')}")

            print("\n--- Test Query 2 (After 'No', providing details) ---")
            query2 = "When I try to turn it on, it just shows the Apple logo for a few seconds then goes black."
            print(f"User: {query2}")
            response2 = controller.process_query(convo_id, query2)
            print(f"Agent:\n{response2.get('reply')}")
            print(f"Ask Feedback: {response2.get('ask_feedback')}")

            if response2.get('ask_feedback'):
                 print("\n--- Test Feedback (Scenario: Yes) ---")
                 feedback_yes_response = controller.process_feedback(convo_id, success=True)
                 print(f"User: Yes")
                 print(f"Agent:\n{feedback_yes_response.get('reply')}")
                 print(f"Reward Awarded: {feedback_yes_response.get('reward')}")

        print("\n--- Test Query 4: Informational ---")
        convo_id_info = f"TEST_INFO_{datetime.now().strftime('%H%M%S')}"
        query4 = "Where can I find the user manual for the DesignPro software?"
        print(f"User: {query4}")
        response4 = controller.process_query(convo_id_info, query4)
        print(f"Agent:\n{response4.get('reply')}")
        print(f"Ask Feedback: {response4.get('ask_feedback')}")

        print("\n--- Test Query 5: Out of Scope ---")
        convo_id_oos = f"TEST_OOS_{datetime.now().strftime('%H%M%S')}"
        query5 = "Can you recommend a good recipe for chocolate chip cookies?"
        print(f"User: {query5}")
        response5 = controller.process_query(convo_id_oos, query5)
        print(f"Agent:\n{response5.get('reply')}")

        print("\n--- Test Query 6: Vague Initial Query ---")
        convo_id_vague = f"TEST_VAGUE_{datetime.now().strftime('%H%M%S')}"
        query6 = "it doesnt work"
        print(f"User: {query6}")
        response6 = controller.process_query(convo_id_vague, query6)
        print(f"Agent:\n{response6.get('reply')}")
        print(f"Ask Feedback: {response6.get('ask_feedback')}")

    except FileNotFoundError as fnf_error:
         logging.critical(f"TESTING FAILED: Could not find CSV file. {fnf_error}", exc_info=False)
         sys.exit(1)
    except ValueError as val_error:
         logging.critical(f"TESTING FAILED: Value error during initialization or processing. {val_error}", exc_info=True)
         sys.exit(1)
    except ConnectionError as conn_error:
         logging.critical(f"TESTING FAILED: Ollama connection error. Is Ollama running and model '{TEXT_MODEL}' available? {conn_error}", exc_info=False)
         sys.exit(1)
    except RuntimeError as rt_error:
         logging.critical(f"TESTING FAILED: Ollama runtime error during processing. {rt_error}", exc_info=True)
         sys.exit(1)
    except Exception as e:
         logging.critical(f"TESTING FAILED: An unexpected critical error occurred. {e}", exc_info=True)
         sys.exit(1)

    logging.info("--- backend_logic.py testing script finished ---")