import os
from typing import Optional
from anthropic import Anthropic
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OutreachPlan:
    collaborators: str
    messages: str
    action_plan: str
    timestamp: datetime = datetime.now()

class BusinessOutreachAssistant:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the assistant with Anthropic API key."""
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        if not api_key and not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError(("No API key provided. Set ANTHROPIC_API_KEY "
                              " environment variable or pass key to constructor."))

    def call_claude(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Make a call to Claude API with error handling."""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Anthropic.APIError as e:
            raise Exception(f"API call failed: {str(e)}")

    def generate_outreach_plan(self, business_idea: str) -> OutreachPlan:
        """Generate a complete outreach plan for a business idea."""
        
        # 1. Generate potential collaborators
        prompt_collaborators = f"""
        Please list at least five potential partners or organizations that
        could help grow this business idea: {business_idea}
        
        For each suggestion, provide:
        - Name of organization/partner
        - Brief rationale for partnership
        - Potential value proposition
        """
        collaborators_list = self.call_claude(prompt_collaborators)

        # 2. Create personalized outreach messages
        prompt_outreach = f"""
        Create professional and engaging introductory emails for these
        potential collaborators:
        {collaborators_list}

        For each email:
        - Use appropriate formal tone
        - Briefly explain the business concept
        - Highlight specific mutual benefits
        - Include clear call to action
        - Keep it concise but compelling
        """
        outreach_messages = self.call_claude(prompt_outreach)

        # 3. Generate action plan
        prompt_next_steps = f"""
        Based on these potential collaborators:
        {collaborators_list}

        And these drafted messages:
        {outreach_messages}

        Please provide a strategic action plan covering:
        1. Prioritized outreach sequence with rationale
        2. Recommended tracking system for responses
        3. Key metrics to measure outreach success
        4. Follow-up strategy
        5. Timeline recommendations
        """
        action_plan = self.call_claude(prompt_next_steps)

        return OutreachPlan(
            collaborators=collaborators_list,
            messages=outreach_messages,
            action_plan=action_plan
        )

def main():
    # Example usage
    business_idea = ("A mobile app for scheduling short, on-demand online "
        "lessons with experts.")
    
    try:
        assistant = BusinessOutreachAssistant()
        plan = assistant.generate_outreach_plan(business_idea)
        
        # Print results with clear formatting
        print("\n=== POTENTIAL COLLABORATORS ===")
        print(plan.collaborators)
        
        print("\n=== OUTREACH MESSAGES ===")
        print(plan.messages)
        
        print("\n=== ACTION PLAN ===")
        print(plan.action_plan)
        
        # Optionally save results
        timestamp = plan.timestamp.strftime("%Y%m%d_%H%M%S")
        with open(f"outreach_plan_{timestamp}.txt", "w") as f:
            f.write(f"Business Idea: {business_idea}\n\n")
            f.write(f"Generated on: {plan.timestamp}\n\n")
            f.write("=== POTENTIAL COLLABORATORS ===\n")
            f.write(plan.collaborators + "\n\n")
            f.write("=== OUTREACH MESSAGES ===\n")
            f.write(plan.messages + "\n\n")
            f.write("=== ACTION PLAN ===\n")
            f.write(plan.action_plan)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()