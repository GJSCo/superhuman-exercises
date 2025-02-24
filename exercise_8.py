import os
import time
import logging
from typing import List, Dict, Any, Optional, Union

class InterviewSchedulerAgent:
    """
    An autonomous AI agent responsible for scheduling interviews.
    This agent handles the entire workflow from processing scheduling requests
    to generating appropriate scheduling suggestions and confirmation messages.
    """
    
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "gpt-4o", 
        log_level: str = "INFO"
    ):
        """
        Initialize the InterviewSchedulerAgent.
        
        Args:
            api_key: OpenAI API key. If None, tries to get from environment
                     variables.
            model: The LLM model to use for generating suggestions.
            log_level: Logging level (INFO, DEBUG, WARNING, ERROR).
        """
        # Setup logging
        self.logger = logging.getLogger("InterviewSchedulerAgent")
        self.logger.setLevel(getattr(logging, log_level))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.warning(
                "No API key provided. Please set it via constructor or "
                "OPENAI_API_KEY environment variable."
            )
        
        # Initialize OpenAI client
        try:
            import openai
            self.openai_client = openai.OpenAI(api_key=self.api_key)
            self.logger.info("OpenAI client initialized successfully.")
        except ImportError:
            self.logger.error(
                "Failed to import openai. Please install it with 'pip install openai'."
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
        
        # Agent configuration
        self.model = model
        self.banned_phrases = [
            "discriminate", "private_info", "confidential", 
            "ssn", "credit card"
        ]
        self.logger.info(
            f"InterviewSchedulerAgent initialized with model: {self.model}"
        )
    
    def generate_scheduling_suggestions(
        self, 
        request_info: Dict[str, Any]
    ) -> str:
        """
        Generate scheduling suggestions based on the candidate's preferences.
        
        Args:
            request_info: Dictionary containing candidate information and preferences.
                Must include 'candidate_name', 'preferred_time_slots', 'timezone'.
                
        Returns:
            String containing suggested interview times.
        """
        self.logger.debug(
            f"Generating scheduling suggestions for: {request_info['candidate_name']}"
        )
        
        # Validate request information
        required_fields = ['candidate_name', 'preferred_time_slots', 'timezone']
        for field in required_fields:
            if field not in request_info:
                self.logger.error(f"Missing required field: {field}")
                raise ValueError(
                    f"Missing required field in request_info: {field}"
                )
        
        # Create prompt for the LLM
        prompt = f"""
        Please suggest 3 suitable interview times for a candidate named \
{request_info['candidate_name']} 
        who is available during {request_info['preferred_time_slots']} in \
{request_info['timezone']} time zone.
        Format the times clearly with date, time, and timezone.
        Thank you.
        """
        
        try:
            # Generate suggestions using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            suggestions = response.choices[0].message.content.strip()
            self.logger.debug(f"Generated suggestions: {suggestions}")
            return suggestions
        except Exception as e:
            self.logger.error(
                f"Error generating scheduling suggestions: {str(e)}"
            )
            raise
    
    def ethical_check(self, text: str) -> bool:
        """
        Perform an ethical check on the generated text.
        
        Args:
            text: The text to check for ethical compliance.
            
        Returns:
            Boolean indicating whether the text passes the ethical check.
        """
        self.logger.debug(f"Performing ethical check on text: {text[:50]}...")
        
        # Check for banned phrases
        for phrase in self.banned_phrases:
            if phrase.lower() in text.lower():
                self.logger.warning(
                    f"Ethical check failed: Found banned phrase '{phrase}'"
                )
                return False
        
        # Additional checks could be implemented here
        self.logger.debug("Ethical check passed")
        return True
    
    def generate_confirmation_message(
        self, 
        candidate_name: str, 
        suggestions: str
    ) -> str:
        """
        Generate a confirmation message for the candidate.
        
        Args:
            candidate_name: Name of the candidate.
            suggestions: The suggested interview times.
            
        Returns:
            A confirmation message string.
        """
        self.logger.debug(f"Generating confirmation message for: {candidate_name}")
        
        confirmation_prompt = f"""
        Hello, please create a polite confirmation message for {candidate_name} 
        using the following suggested interview times: 
        {suggestions}
        
        The message should ask them to select their preferred time and respond to confirm.
        Thank you.
        """
        
        try:
            # Generate confirmation message using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": confirmation_prompt}],
                max_tokens=150
            )
            confirmation_msg = response.choices[0].message.content.strip()
            self.logger.debug(
                f"Generated confirmation message: {confirmation_msg[:50]}..."
            )
            return confirmation_msg
        except Exception as e:
            self.logger.error(
                f"Error generating confirmation message: {str(e)}"
            )
            raise
    
    def process_interview_requests(
        self, 
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple interview scheduling requests.
        
        Args:
            requests: A list of dictionaries containing candidate information and preferences.
            
        Returns:
            A list of dictionaries containing scheduling results or errors.
        """
        self.logger.info(f"Processing {len(requests)} interview requests")
        results = []
        
        for req in requests:
            try:
                self.logger.info(
                    f"Processing request for candidate: "
                    f"{req.get('candidate_name', 'Unknown')}"
                )
                
                # Step 1: Generate scheduling suggestions
                suggestions = self.generate_scheduling_suggestions(req)
                
                # Step 2: Perform ethical check
                if self.ethical_check(suggestions):
                    # Step 3: Generate confirmation message
                    confirmation_msg = self.generate_confirmation_message(
                        req["candidate_name"], 
                        suggestions
                    )
                    
                    # Add result to the list
                    results.append({
                        "candidate": req["candidate_name"],
                        "suggested_times": suggestions,
                        "confirmation_message": confirmation_msg,
                        "status": "success"
                    })
                    self.logger.info(
                        f"Successfully processed request for: "
                        f"{req['candidate_name']}"
                    )
                else:
                    # Handle ethical check failure
                    results.append({
                        "candidate": req["candidate_name"],
                        "error": "Scheduling suggestions did not pass the ethical check.",
                        "status": "failed"
                    })
                    self.logger.warning(
                        f"Ethical check failed for candidate: "
                        f"{req['candidate_name']}"
                    )
            except Exception as e:
                # Handle other errors
                results.append({
                    "candidate": req.get("candidate_name", "Unknown"),
                    "error": str(e),
                    "status": "error"
                })
                self.logger.error(f"Error processing request: {str(e)}")
        
        self.logger.info(
            f"Completed processing {len(requests)} requests with "
            f"{sum(1 for r in results if r.get('status') == 'success')} successes"
        )
        return results
    
    def update_banned_phrases(
        self, 
        phrases: List[str], 
        append: bool = True
    ) -> None:
        """
        Update the list of banned phrases for ethical checks.
        
        Args:
            phrases: List of phrases to ban.
            append: If True, append to existing list. If False, replace the list.
        """
        if not append:
            self.banned_phrases = phrases
            self.logger.info(
                f"Replaced banned phrases list with {len(phrases)} phrases"
            )
        else:
            self.banned_phrases.extend(phrases)
            self.logger.info(
                f"Added {len(phrases)} phrases to banned phrases list"
            )


# Example usage
if __name__ == "__main__":
    # Create the agent
    agent = InterviewSchedulerAgent()
    
    # Example interview requests
    interview_requests = [
        {
            "candidate_name": "Alice",
            "preferred_time_slots": "9am-12pm, 2pm-4pm",
            "timezone": "EST"
        },
        {
            "candidate_name": "Bob",
            "preferred_time_slots": "Anytime Tuesday or Thursday",
            "timezone": "PST"
        }
    ]
    
    # Process the requests
    results = agent.process_interview_requests(interview_requests)
    
    # Print results
    for res in results:
        print(res)
        print("----")