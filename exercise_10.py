from openai import OpenAI
import anthropic
import json
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ================
# Data Models
# ================
class BusinessOpportunity(BaseModel):
    """
    Represents a potential business opportunity identified by the agent.
    
    Attributes:
        name: The name or description of the opportunity
        estimated_roi: Expected return on investment as a percentage
        popularity_score: How popular/trending this opportunity is (1-10)
        ethical_score: How well it aligns with ethical guidelines (1-10)
    """
    name: str = Field(..., description="Name or description of the business opportunity")
    estimated_roi: float = Field(..., description="Expected ROI percentage")
    popularity_score: float = Field(..., description="Popularity score from 1-10")
    ethical_score: Optional[float] = Field(None, description="Ethical alignment score from 1-10")


class ToolAction(BaseModel):
    """
    Represents a tool action decision made by the LLM.
    
    Attributes:
        tool_name: The name of the tool to use
        reason: The reasoning for using this tool
        parameters: Parameters to pass to the tool
    """
    tool_name: str = Field(..., description="Name of the tool to use")
    reason: str = Field(..., description="Reasoning for using this tool")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the tool")


class OpportunityWithActions(BaseModel):
    """
    Combines a business opportunity with the actions to be taken on it.
    
    Attributes:
        opportunity: The opportunity information
        actions: List of tool actions to perform on this opportunity
    """
    opportunity: BusinessOpportunity = Field(..., description="The business opportunity")
    actions: List[ToolAction] = Field(..., description="List of actions to perform on this opportunity")


class LLMResponse(BaseModel):
    """
    Structured response from the LLM analysis.
    
    Attributes:
        summary: A brief summary of the analysis
        opportunities_with_actions: A list of identified opportunities with tool actions
    """
    summary: str = Field(..., description="Brief summary of the analysis")
    opportunities_with_actions: List[OpportunityWithActions] = Field(default_factory=list, 
                                                                    description="List of opportunities with actions")


class ToolResult(BaseModel):
    """
    Standard response format for any tool execution.
    
    Attributes:
        success: Whether the tool executed successfully
        message: Information about the execution result
        data: Optional additional data returned by the tool
    """
    success: bool = Field(..., description="Whether the tool execution succeeded")
    message: str = Field(..., description="Information about the execution result")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data returned by the tool")


# ================
# Identity and Goals
# ================
class IdentityContext:
    def __init__(self, name, domain_expertise, ethical_guidelines):
        self.name = name
        self.domain_expertise = domain_expertise
        self.ethical_guidelines = ethical_guidelines

    def get_identity_profile(self):
        return {
            "agent_name": self.name,
            "expertise": self.domain_expertise,
            "ethical_guidelines": self.ethical_guidelines
        }


class GoalFramework:
    def __init__(self, primary_goal, secondary_goals, constraints):
        self.primary_goal = primary_goal
        self.secondary_goals = secondary_goals
        self.constraints = constraints

    def get_goals(self):
        return {
            "primary_goal": self.primary_goal,
            "secondary_goals": self.secondary_goals,
            "constraints": self.constraints
        }


# ================
# LLM Service
# ================
class IntelligenceLayer:
    """
    Service for interacting with Language Model APIs.
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI()  # Initialize the client
        
        # Check for alternative providers
        self.anthropic_available = os.environ.get("ANTHROPIC_API_KEY") is not None
        if self.anthropic_available:
            self.anthropic_client = anthropic.Anthropic()
    
    def generate_response(self, prompt, temperature=0.7):
        """Simple response generation without structured output."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    def analyze_opportunities(self, user_prompt, identity, goals, available_tools):
        """
        Analyzes user request to identify opportunities and suggest tools.
        
        Args:
            user_prompt: The user's business request
            identity: The agent's identity context
            goals: The agent's goal framework
            available_tools: List of available tool names with descriptions
            
        Returns:
            A structured LLMResponse with summary and opportunities_with_actions
        """
        # Get identity and goals
        identity_profile = identity.get_identity_profile()
        goals_profile = goals.get_goals()
        
        # Generate the system instructions
        system_instructions = (
            f"Agent Name: {identity_profile['agent_name']}\n"
            f"Expertise: {identity_profile['expertise']}\n"
            f"Ethical Guidelines: {identity_profile['ethical_guidelines']}\n\n"
            f"Primary Goal: {goals_profile['primary_goal']}\n"
            f"Secondary Goals: {', '.join(goals_profile['secondary_goals'])}\n"
            f"Constraints: {', '.join(goals_profile['constraints'])}\n\n"
        )
        
        # Generate tool information
        tool_info = "\n".join([f"- {name}: {desc}" for name, desc in available_tools.items()])
        
        # Check if Anthropic is available, otherwise use OpenAI
        if self.anthropic_available:
            return self._call_anthropic_api(system_instructions, user_prompt, tool_info)
        else:
            return self._call_openai_api(system_instructions, user_prompt, tool_info)
    
    def _call_openai_api(self, system_instructions, user_prompt, tool_info):
        """Use OpenAI to analyze opportunities and suggest tools."""
        system_prompt = f"""
        {system_instructions}
        
        You are an AI assistant that analyzes business requests to identify viable opportunities.
        
        You have the following tools available:
        {tool_info}
        
        Always respond in valid JSON format with a 'summary' field and an 'opportunities_with_actions' array.
        Each opportunity should have:
        1. An 'opportunity' object with 'name', 'estimated_roi', 'popularity_score', and 'ethical_score'
        2. An 'actions' array listing which tools to use, why, and with what parameters
        """
        
        user_message = f"""
        Analyze this business request and extract:
        1. A brief summary of the request (1-2 sentences)
        2. A list of viable business opportunities
        3. For each opportunity, decide which tools to use for further analysis

        For each opportunity, decide which tools to use based on these criteria:
        - The market_research tool should be used when you need more detailed market information
        - The roi_calculator tool should be used to get precise ROI estimates
        - The ethical_evaluator tool should be used when ethical considerations are important

        Respond ONLY with valid JSON matching this structure:
        {{
            "summary": "...",
            "opportunities_with_actions": [
                {{
                    "opportunity": {{
                        "name": "...", 
                        "estimated_roi": 0.15,
                        "popularity_score": 8.5,
                        "ethical_score": 7.0
                    }},
                    "actions": [
                        {{
                            "tool_name": "market_research|roi_calculator|ethical_evaluator",
                            "reason": "Why this tool should be used",
                            "parameters": {{
                                "query": "market research query" // or other parameters depending on the tool
                            }}
                        }},
                        // More actions if needed
                    ]
                }},
                // More opportunities if needed
            ]
        }}
        
        User's request: {user_prompt}
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0.2
        )
        
        content = response.choices[0].message.content
        
        # Parse the response into our data model
        try:
            data = json.loads(content)
            
            # Convert to our data model
            opportunities_with_actions = []
            
            for opp_entry in data.get("opportunities_with_actions", []):
                opp_data = opp_entry.get("opportunity", {})
                actions_data = opp_entry.get("actions", [])
                
                opportunity = BusinessOpportunity(
                    name=opp_data.get("name", "Unnamed opportunity"),
                    estimated_roi=opp_data.get("estimated_roi", 0.0),
                    popularity_score=opp_data.get("popularity_score", 5.0),
                    ethical_score=opp_data.get("ethical_score")
                )
                
                actions = [
                    ToolAction(
                        tool_name=action.get("tool_name"),
                        reason=action.get("reason", "No reason provided"),
                        parameters=action.get("parameters")
                    ) for action in actions_data
                ]
                
                opportunities_with_actions.append(OpportunityWithActions(
                    opportunity=opportunity, 
                    actions=actions
                ))
            
            return LLMResponse(
                summary=data.get("summary", "No summary provided"),
                opportunities_with_actions=opportunities_with_actions
            )
        
        except Exception as e:
            # In case of parsing error, return a minimal valid response
            return LLMResponse(
                summary=f"Error parsing response: {str(e)}",
                opportunities_with_actions=[]
            )
    
    def _call_anthropic_api(self, system_instructions, user_prompt, tool_info):
        """Use Anthropic Claude to analyze opportunities and suggest tools."""
        # System and user prompts for Claude
        system_prompt = f"""
        {system_instructions}
        
        You are an AI assistant that analyzes business requests to identify viable opportunities.
        
        You have the following tools available:
        {tool_info}
        """
        
        user_message = f"""
        Analyze this business request and extract:
        1. A brief summary of the request (1-2 sentences)
        2. A list of viable business opportunities
        3. For each opportunity, decide which tools to use for further analysis

        For each opportunity, decide which tools to use based on these criteria:
        - The market_research tool should be used when you need more detailed market information
        - The roi_calculator tool should be used to get precise ROI estimates
        - The ethical_evaluator tool should be used when ethical considerations are important

        Respond ONLY with valid JSON matching this structure:
        {{
            "summary": "...",
            "opportunities_with_actions": [
                {{
                    "opportunity": {{
                        "name": "...", 
                        "estimated_roi": 0.15,
                        "popularity_score": 8.5,
                        "ethical_score": 7.0
                    }},
                    "actions": [
                        {{
                            "tool_name": "market_research|roi_calculator|ethical_evaluator",
                            "reason": "Why this tool should be used",
                            "parameters": {{
                                "query": "market research query" // or other parameters depending on the tool
                            }}
                        }}
                    ]
                }}
            ]
        }}
        
        User's request: {user_prompt}
        """
        
        # Make the API request to Anthropic
        response = self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            max_tokens=4096,
            temperature=0.2,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        # Extract content from the response
        content = response.content[0].text
        
        # Extract JSON from the response (Claude might add additional text)
        import re
        json_match = re.search(r'(\{[\s\S]*\})', content)
        if not json_match:
            raise Exception("Could not extract JSON from Claude's response")
        
        json_content = json_match.group(1)
        
        # Parse the response into our data model
        try:
            data = json.loads(json_content)
            
            # Convert to our data model
            opportunities_with_actions = []
            
            for opp_entry in data.get("opportunities_with_actions", []):
                opp_data = opp_entry.get("opportunity", {})
                actions_data = opp_entry.get("actions", [])
                
                opportunity = BusinessOpportunity(
                    name=opp_data.get("name", "Unnamed opportunity"),
                    estimated_roi=opp_data.get("estimated_roi", 0.0),
                    popularity_score=opp_data.get("popularity_score", 5.0),
                    ethical_score=opp_data.get("ethical_score")
                )
                
                actions = [
                    ToolAction(
                        tool_name=action.get("tool_name"),
                        reason=action.get("reason", "No reason provided"),
                        parameters=action.get("parameters")
                    ) for action in actions_data
                ]
                
                opportunities_with_actions.append(OpportunityWithActions(
                    opportunity=opportunity, 
                    actions=actions
                ))
            
            return LLMResponse(
                summary=data.get("summary", "No summary provided"),
                opportunities_with_actions=opportunities_with_actions
            )
        
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response: {json_content}")
        except Exception as e:
            # In case of parsing error, return a minimal valid response
            return LLMResponse(
                summary=f"Error parsing response: {str(e)}",
                opportunities_with_actions=[]
            )


# ================
# Base Tool
# ================
class BaseTool:
    """
    Base class for all tools that agents can use.
    """
    name: str = "base_tool"
    description: str = "Base tool class that all tools derive from."
    
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool's functionality."""
        raise NotImplementedError("Derived tools must implement the 'run' method")


# ================
# Market Research Tool
# ================
class MarketResearchTool(BaseTool):
    """Tool for researching market opportunities."""
    name: str = "market_research"
    description: str = "Researches market trends and opportunities"
    
    def run(self, query: str) -> ToolResult:
        """Research market opportunities based on query."""
        try:
            # Mock: In reality, integrate an API call or a web-scraper
            results = [
                {"opportunity": "Dropshipping in specialized niche", "popularity_score": 7.5},
                {"opportunity": "Affiliate marketing for eco-friendly products", "popularity_score": 8.2},
                {"opportunity": "Online course in specialized skill", "popularity_score": 6.8}
            ]
            
            # Filter based on query if needed
            if "eco" in query.lower():
                results = [r for r in results if "eco" in r["opportunity"].lower()]
            elif "course" in query.lower():
                results = [r for r in results if "course" in r["opportunity"].lower()]
                
            return ToolResult(
                success=True,
                message=f"Successfully researched market for: {query}",
                data={"query": query, "results": results}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Failed to research market: {str(e)}"
            )


# ================
# ROI Calculator Tool
# ================
class ROICalculatorTool(BaseTool):
    """Tool for calculating return on investment."""
    name: str = "roi_calculator"
    description: str = "Calculates expected ROI for business opportunities"
    
    def run(self, business_model: str) -> ToolResult:
        """Calculate ROI for a specific business model."""
        try:
            # Mock: returns a simplified ROI estimate
            roi_map = {
                "Dropshipping in specialized niche": 0.15,
                "Affiliate marketing for eco-friendly products": 0.20,
                "Online course in specialized skill": 0.10
            }
            
            roi = roi_map.get(business_model, 0.0)
            if roi == 0.0 and business_model:
                # Generate fallback ROI for unknown models
                if "dropshipping" in business_model.lower():
                    roi = 0.12
                elif "affiliate" in business_model.lower():
                    roi = 0.18
                elif "course" in business_model.lower():
                    roi = 0.09
                else:
                    roi = 0.05
                
            return ToolResult(
                success=True,
                message=f"Calculated ROI for: {business_model}",
                data={"business_model": business_model, "estimated_roi": roi}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Failed to calculate ROI: {str(e)}"
            )


# ================
# Ethical Evaluator Tool
# ================
class EthicalEvaluatorTool(BaseTool):
    """Tool for evaluating the ethical implications of business opportunities."""
    name: str = "ethical_evaluator"
    description: str = "Evaluates ethical implications of business opportunities"
    
    def run(self, business_model: str) -> ToolResult:
        """Evaluate ethical considerations for a business model."""
        try:
            # Mock: returns a simplified ethical score
            ethical_map = {
                "Dropshipping in specialized niche": 6.5,
                "Affiliate marketing for eco-friendly products": 8.5,
                "Online course in specialized skill": 9.0
            }
            
            ethical_score = ethical_map.get(business_model, 5.0)
            if ethical_score == 5.0 and business_model:
                # Generate fallback ethical score for unknown models
                if "eco" in business_model.lower() or "sustainable" in business_model.lower():
                    ethical_score = 8.0
                elif "education" in business_model.lower() or "course" in business_model.lower():
                    ethical_score = 8.5
                
            return ToolResult(
                success=True,
                message=f"Evaluated ethical considerations for: {business_model}",
                data={"business_model": business_model, "ethical_score": ethical_score}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Failed to evaluate ethics: {str(e)}"
            )


# ================
# Execution Engine
# ================
class ExecutionEngine:
    """
    Main engine that orchestrates the AI agent's workflow.
    
    Handles the flow from user input to final recommendations.
    """
    def __init__(self, intelligence_layer, identity, goals, tools):
        self.intelligence = intelligence_layer
        self.identity = identity
        self.goals = goals
        self.tools = tools
        self.memory = []
    
    def run(self, user_prompt):
        """
        Process the user's prompt and return business recommendations.
        
        Args:
            user_prompt: The business request from the user
            
        Returns:
            Final recommendations and analysis
        """
        print("\n------ Processing Business Request ------")
        
        # Get list of available tools with descriptions
        available_tools = {tool.name: tool.description for tool in self.tools.values()}
        
        # Use the LLM to identify opportunities and determine which tools to use
        llm_response = self.intelligence.analyze_opportunities(
            user_prompt, 
            self.identity, 
            self.goals, 
            available_tools
        )
        
        # Save initial analysis to memory
        self.memory.append({"initial_analysis": llm_response.summary})
        
        # Output the summary
        print("\n------ AI Summary ------")
        print(llm_response.summary)
        
        # Process the opportunities and execute the tool actions
        tool_actions_results = []
        enhanced_opportunities = []
        
        if not llm_response.opportunities_with_actions:
            print("\nNo viable opportunities identified.")
        else:
            print(f"\n------ Found {len(llm_response.opportunities_with_actions)} Opportunities ------")
            
            for i, opp_with_actions in enumerate(llm_response.opportunities_with_actions, 1):
                opportunity = opp_with_actions.opportunity
                actions = opp_with_actions.actions
                
                print(f"\nOpportunity {i}: {opportunity.name}")
                print(f"Estimated ROI: {opportunity.estimated_roi:.1%}")
                print(f"Popularity Score: {opportunity.popularity_score}/10")
                if opportunity.ethical_score:
                    print(f"Ethical Score: {opportunity.ethical_score}/10")
                
                enhanced_opportunity = opportunity.model_copy()
                action_results = []
                
                print("\nActions selected by AI:")
                for action in actions:
                    print(f"- Tool: {action.tool_name}")
                    print(f"  Reason: {action.reason}")
                    if action.parameters:
                        parameters_str = ", ".join([f"{k}={v}" for k, v in action.parameters.items()])
                        print(f"  Parameters: {parameters_str}")
                    
                    # Execute the tool based on the LLM's decision
                    tool = self.get_tool(action.tool_name)
                    if not tool:
                        print(f"  Error: Tool '{action.tool_name}' not found")
                        continue
                    
                    # Call the appropriate tool with the right parameters
                    if action.tool_name == "market_research":
                        query = action.parameters.get("query", opportunity.name)
                        result = tool.run(query=query)
                    elif action.tool_name == "roi_calculator":
                        result = tool.run(business_model=opportunity.name)
                        # Update ROI if successful
                        if result.success and result.data and "estimated_roi" in result.data:
                            enhanced_opportunity.estimated_roi = result.data["estimated_roi"]
                    elif action.tool_name == "ethical_evaluator":
                        result = tool.run(business_model=opportunity.name)
                        # Update ethical score if successful
                        if result.success and result.data and "ethical_score" in result.data:
                            enhanced_opportunity.ethical_score = result.data["ethical_score"]
                    else:
                        # Generic parameter passing
                        result = tool.run(**(action.parameters or {}))
                    
                    action_results.append({
                        "tool": action.tool_name,
                        "parameters": action.parameters,
                        "result": result.model_dump() if hasattr(result, "model_dump") else result
                    })
                
                enhanced_opportunities.append(enhanced_opportunity)
                tool_actions_results.append({
                    "opportunity": opportunity.model_dump(),
                    "actions_results": action_results
                })
        
        # Store results in memory
        self.memory.append({"tool_executions": tool_actions_results})
        
        # Generate final recommendations using tool results
        final_recommendations = self.generate_final_recommendations(
            user_prompt, 
            llm_response.summary, 
            enhanced_opportunities
        )
        
        return final_recommendations
    
    def get_tool(self, tool_name):
        """Get a tool by name from the available tools."""
        return self.tools.get(tool_name)
    
    def generate_final_recommendations(self, user_prompt, initial_summary, opportunities):
        """Generate final recommendations based on analysis and tool results."""
        identity_profile = self.identity.get_identity_profile()
        goals_profile = self.goals.get_goals()
        
        # Create a prompt for the final recommendations
        prompt = f"""
        Agent Name: {identity_profile['agent_name']}
        Expertise: {identity_profile['expertise']}
        Ethical Guidelines: {identity_profile['ethical_guidelines']}
        
        Primary Goal: {goals_profile['primary_goal']}
        
        User Request: {user_prompt}
        
        Initial Analysis: {initial_summary}
        
        Analyzed Opportunities:
        """
        
        # Add each opportunity with its updated information
        for i, opp in enumerate(opportunities, 1):
            prompt += f"""
            {i}. {opp.name}
               - Estimated ROI: {opp.estimated_roi:.1%}
               - Popularity Score: {opp.popularity_score}/10
               - Ethical Score: {opp.ethical_score if opp.ethical_score else 'Not evaluated'}/10
            """
        
        prompt += """
        Based on the above information, please provide:
        1. A summary of the most promising opportunities
        2. Specific recommendations on which opportunity(s) to pursue
        3. Next steps for implementation
        4. Any ethical considerations or constraints to be aware of
        
        Focus on being practical, specific, and actionable.
        """
        
        # Generate final recommendations
        final_recommendations = self.intelligence.generate_response(prompt, temperature=0.5)
        self.memory.append({"final_recommendations": final_recommendations})
        
        return final_recommendations


def main():
    # Instantiate components
    model_name = "gpt-4o"  # Example model
    intelligence_layer = IntelligenceLayer(model_name=model_name)

    identity = IdentityContext(
        name="Entrepreneur Agent",
        domain_expertise="Business development, market analysis, revenue strategy",
        ethical_guidelines="Focus on fair, responsible, and community-friendly opportunities."
    )

    goals = GoalFramework(
        primary_goal="Generate a list of viable new revenue streams",
        secondary_goals=[
            "Assess ethical impact",
            "Estimate potential ROI",
            "Propose next action steps"
        ],
        constraints=[
            "Respect user's ethical guidelines",
            "All suggestions must be legal and community-friendly"
        ]
    )

    # Create tools
    market_tool = MarketResearchTool()
    roi_tool = ROICalculatorTool()
    ethical_tool = EthicalEvaluatorTool()

    tools = {
        market_tool.name: market_tool,
        roi_tool.name: roi_tool,
        ethical_tool.name: ethical_tool
    }

    engine = ExecutionEngine(
        intelligence_layer=intelligence_layer,
        identity=identity,
        goals=goals,
        tools=tools
    )

    # Sample user prompt
    user_prompt = (
        "Hello Entrepreneur Agent. Please suggest some new revenue streams for "
        "an online educator who wants to diversify income. Thank you!"
    )

    # Run the agent
    output = engine.run(user_prompt)
    print("\n--- FINAL RECOMMENDATIONS ---")
    print(output)

if __name__ == "__main__":
    main()