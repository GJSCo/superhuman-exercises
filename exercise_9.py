import json
import time
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Union, Set
from pydantic import BaseModel, Field, ConfigDict


# ================
# Data Models
# ================
class Task(BaseModel):
    """
    Represents a unit of work discovered by an LLM or
    any other system that identifies tasks in user input.
    
    Attributes:
        name: The name or description of the task
        deadline: The deadline for the task in ISO format (YYYY-MM-DD)
        priority: Optional priority level of the task
        assignee: Optional person assigned to the task
    """
    name: str = Field(..., description="Name or description of the task")
    deadline: str = Field(..., description="Deadline for the task in ISO format (YYYY-MM-DD)")
    priority: Optional[str] = Field(None, description="Priority level (high, medium, low)")
    assignee: Optional[str] = Field(None, description="Person assigned to the task")
    
    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True
    )
    
    def is_deadline_realistic(self) -> bool:
        """
        Determine if the deadline is realistic based on current date.
        A deadline is considered unrealistic if it's in the past or less than 24 hours away.
        
        Returns:
            bool: True if the deadline is realistic, False otherwise
        """
        deadline_date = datetime.fromisoformat(self.deadline)
        today = datetime.now()
        
        # Consider a deadline unrealistic if it's in the past or less than 24 hours away
        return deadline_date >= today and (deadline_date - today).days >= 1


class ToolAction(BaseModel):
    """
    Represents a tool action decision made by the LLM.
    
    Attributes:
        tool_name: The name of the tool to use
        reason: The reasoning for using this tool
        message: Optional custom message to use with the tool
    """
    tool_name: str = Field(..., description="Name of the tool to use")
    reason: str = Field(..., description="Reasoning for using this tool")
    message: Optional[str] = Field(None, description="Custom message to use with the tool")


class TaskWithActions(BaseModel):
    """
    Combines a task with the actions to be taken on it.
    
    Attributes:
        task: The task information
        actions: List of tool actions to perform on this task
    """
    task: Task = Field(..., description="The task information")
    actions: List[ToolAction] = Field(..., description="List of actions to perform on this task")


class LLMResponse(BaseModel):
    """
    Structured response from the LLM parsing function.
    
    Attributes:
        summary: A brief summary of the parsed text
        tasks_with_actions: A list of identified tasks with associated tool actions
    """
    summary: str = Field(..., description="Brief summary of the parsed text")
    tasks_with_actions: List[TaskWithActions] = Field(default_factory=list, description="List of tasks with associated actions")
    
    model_config = ConfigDict(
        extra="ignore"
    )


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
# LLM Service
# ================
class LLMService:
    """
    Service for interacting with Language Model APIs.
    
    This service encapsulates the complexity of making API calls 
    to various LLM providers and parsing their responses.
    """
    
    @staticmethod
    def call(prompt: str, available_tools: Optional[List[str]] = None) -> LLMResponse:
        """
        Makes a call to an LLM API to process the text.
        
        Args:
            prompt: The prompt to send to the LLM
            available_tools: List of available tool names the LLM can choose from
            
        Returns:
            A structured LLMResponse with summary and tasks_with_actions
        """
        # Extract the text to analyze from the prompt
        import re
        text_match = re.search(r"Text: (.*?)(?:\n\n|$)", prompt, re.DOTALL)
        if not text_match:
            return LLMResponse(summary="No text found to analyze", tasks_with_actions=[])
        
        text_to_analyze = text_match.group(1).strip()
        
        # Check for Anthropic API key first
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if anthropic_api_key:
            return LLMService._call_anthropic_api(anthropic_api_key, text_to_analyze, available_tools)
        
        # Fall back to OpenAI if no Anthropic key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if openai_api_key:
            return LLMService._call_openai_api(openai_api_key, text_to_analyze, available_tools)
        else:
            raise Exception("Neither Anthropic nor OpenAI API keys found in environment variables")
    
    @staticmethod
    def _get_tool_descriptions_and_info(available_tools: Optional[List[str]] = None) -> tuple:
        """
        Gets tool descriptions and formatted info for prompts.
        
        Args:
            available_tools: List of available tool names
            
        Returns:
            Tuple of (tool_descriptions, tool_info_string)
        """
        # Default tools if none provided
        if not available_tools:
            available_tools = ["scheduler", "notifier"]
        
        # Tool descriptions
        tool_descriptions = {
            "scheduler": "Schedules tasks in a calendar system. Use this tool for all identified tasks.",
            "notifier": "Sends notifications about tasks. Use this tool when deadlines are unrealistic, important tasks need attention, or specific individuals need to be informed."
        }
        
        # Generate tool information for the prompt
        tool_info = "\n".join([f"- {tool}: {tool_descriptions.get(tool, 'No description')}" for tool in available_tools])
        
        return tool_descriptions, tool_info

    @staticmethod
    def _call_anthropic_api(api_key: str, text: str, available_tools: Optional[List[str]] = None) -> LLMResponse:
        """
        Makes a call to Anthropic's Claude API.
        
        Args:
            api_key: Anthropic API key
            text: Text to analyze
            available_tools: List of available tool names
            
        Returns:
            LLMResponse with summary and tasks_with_actions
        """
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Get tool descriptions and info
        _, tool_info = LLMService._get_tool_descriptions_and_info(available_tools)
        
        # Construct the prompt for Claude
        system_prompt = f"""You are a task extraction and scheduling assistant. Your job is to analyze text and identify tasks, 
        deadlines, and determine which tools to use.
        
        You have the following tools available:
        {tool_info}
        
        For each task, you must evaluate the deadline and decide which tools are appropriate to use.
        Consider a deadline unrealistic if it's in the past or less than 24 hours away."""
        
        # User prompt with the specific instructions
        user_prompt = f"""Analyze this text and extract:
        1. A brief summary (1-2 sentences)
        2. A list of actionable tasks with deadlines
        3. For each task, decide which tools to use

        Today's Date: {datetime.now().strftime("%Y-%m-%d")}

        For each task, decide which tools to use based on these criteria:
        - The scheduler tool should be used for all tasks
        - The notifier tool should be used when:
          * The deadline is unrealistic (in the past or less than 24 hours away)
          * The task has high priority
          * An explicit notification is needed

        Respond ONLY with valid JSON matching this structure:
        {{
            "summary": "...",
            "tasks_with_actions": [
                {{
                    "task": {{
                        "name": "...", 
                        "deadline": "2025-03-01",
                        "priority": "high|medium|low",  // optional
                        "assignee": "..."  // optional, if mentioned
                    }},
                    "actions": [
                        {{
                            "tool_name": "scheduler|notifier",
                            "reason": "Why this tool should be used",
                            "message": "Custom message if needed" // optional, primarily for notifier
                        }},
                        // More actions if needed
                    ]
                }},
                // More tasks if needed
            ]
        }}
        
        Text to analyze: {text}
        """
        
        # The API request payload
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # Make the API request
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )
        
        # Handle API errors
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        # Parse the response
        result = response.json()
        content = result['content'][0]['text']
        
        # Extract JSON from the response (Claude might add additional text)
        import re
        json_match = re.search(r'(\{[\s\S]*\})', content)
        if not json_match:
            raise Exception("Could not extract JSON from Claude's response")
        
        json_content = json_match.group(1)
        
        # Extract and validate the JSON
        try:
            data = json.loads(json_content)
            
            # Convert to our data model
            tasks_with_actions = []
            
            for task_entry in data.get("tasks_with_actions", []):
                task_data = task_entry.get("task", {})
                actions_data = task_entry.get("actions", [])
                
                task = Task(
                    name=task_data.get("name", "Unnamed task"),
                    deadline=task_data.get("deadline", datetime.now().strftime("%Y-%m-%d")),
                    priority=task_data.get("priority"),
                    assignee=task_data.get("assignee")
                )
                
                actions = [
                    ToolAction(
                        tool_name=action.get("tool_name"),
                        reason=action.get("reason", "No reason provided"),
                        message=action.get("message")
                    ) for action in actions_data
                ]
                
                tasks_with_actions.append(TaskWithActions(task=task, actions=actions))
            
            return LLMResponse(
                summary=data.get("summary", "No summary provided"),
                tasks_with_actions=tasks_with_actions
            )
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response: {json_content}")
        except Exception as e:
            raise Exception(f"Error parsing response: {str(e)}")
    
    @staticmethod
    def _call_openai_api(api_key: str, text: str, available_tools: Optional[List[str]] = None) -> LLMResponse:
        """
        Makes a call to OpenAI's API.
        
        Args:
            api_key: OpenAI API key
            text: Text to analyze
            available_tools: List of available tool names
            
        Returns:
            LLMResponse with summary and tasks_with_actions
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Get tool descriptions and info
        _, tool_info = LLMService._get_tool_descriptions_and_info(available_tools)
        
        # System prompt to define the role and format
        system_prompt = f"""
        You are an AI assistant that analyzes text to extract a summary and actionable tasks.
        
        You have the following tools available:
        {tool_info}
        
        Always respond in valid JSON format with a 'summary' field and a 'tasks_with_actions' array.
        Each task should have:
        1. A 'task' object with 'name', 'deadline' in YYYY-MM-DD format, and optional 'priority' and 'assignee'
        2. An 'actions' array listing which tools to use and why
        
        For each task, you must evaluate the deadline and decide which tools are appropriate to use.
        Consider a deadline unrealistic if it's in the past or less than 24 hours away.
        """
        
        # User prompt with the specific instructions
        user_prompt = f"""
        Analyze this text and extract:
        1. A brief summary (1-2 sentences)
        2. A list of actionable tasks with deadlines
        3. For each task, decide which tools to use

        Today's Date: {datetime.now().strftime("%Y-%m-%d")}

        For each task, decide which tools to use based on these criteria:
        - The scheduler tool should be used for all tasks
        - The notifier tool should be used when:
          * The deadline is unrealistic (in the past or less than 24 hours away)
          * The task has high priority
          * An explicit notification is needed

        Respond ONLY with valid JSON matching this structure:
        {{
            "summary": "...",
            "tasks_with_actions": [
                {{
                    "task": {{
                        "name": "...", 
                        "deadline": "2025-03-01",
                        "priority": "high|medium|low",  // optional
                        "assignee": "..."  // optional, if mentioned
                    }},
                    "actions": [
                        {{
                            "tool_name": "scheduler|notifier",
                            "reason": "Why this tool should be used",
                            "message": "Custom message if needed" // optional, primarily for notifier
                        }},
                        // More actions if needed
                    ]
                }},
                // More tasks if needed
            ]
        }}
        """
        
        # The message array for the chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "\n\nText: " + text}
        ]
        
        # The API request payload
        payload = {
            "model": "gpt-4o",  # or another appropriate model
            "messages": messages,
            "response_format": {"type": "json_object"},  # Force JSON response
            "temperature": 0.2,  # Low temperature for more deterministic output
            "max_tokens": 4096
        }
        
        # Make the API request
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        # Handle API errors
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        # Parse the response
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Extract and validate the JSON
        try:
            data = json.loads(content)
            
            # Convert to our data model
            tasks_with_actions = []
            
            for task_entry in data.get("tasks_with_actions", []):
                task_data = task_entry.get("task", {})
                actions_data = task_entry.get("actions", [])
                
                task = Task(
                    name=task_data.get("name", "Unnamed task"),
                    deadline=task_data.get("deadline", datetime.now().strftime("%Y-%m-%d")),
                    priority=task_data.get("priority"),
                    assignee=task_data.get("assignee")
                )
                
                actions = [
                    ToolAction(
                        tool_name=action.get("tool_name"),
                        reason=action.get("reason", "No reason provided"),
                        message=action.get("message")
                    ) for action in actions_data
                ]
                
                tasks_with_actions.append(TaskWithActions(task=task, actions=actions))
            
            return LLMResponse(
                summary=data.get("summary", "No summary provided"),
                tasks_with_actions=tasks_with_actions
            )
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response: {content}")
        except Exception as e:
            raise Exception(f"Error parsing response: {str(e)}")


# ================
# Base Tool
# ================
class BaseTool:
    """
    Base class for all tools that agents can use.
    
    All derived tools must implement the `run` method.
    """
    name: str = "base_tool"
    description: str = "Base tool class that all tools derive from."
    
    def run(self, **kwargs) -> ToolResult:
        """
        Execute the tool's functionality.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult: The result of the tool execution
            
        Raises:
            NotImplementedError: This method must be implemented by derived classes
        """
        raise NotImplementedError("Derived tools must implement the 'run' method")


# ================
# Scheduler Tool
# ================
class SchedulerTool(BaseTool):
    """
    Tool for scheduling tasks in a calendar system.
    
    In a real implementation, this would integrate with
    calendar APIs like Google Calendar, Microsoft Outlook, etc.
    """
    name: str = "scheduler"
    description: str = "Schedules tasks in a calendar system"
    
    def run(self, task: Task) -> ToolResult:
        """
        Schedule a task in the calendar.
        
        Args:
            task: The task to schedule
            
        Returns:
            ToolResult: The result of the scheduling operation
        """
        # In a real implementation, this would be an API call to a calendar system
        # For demonstration purposes, we just simulate it
        try:
            print(f"[Calendar] Scheduled '{task.name}' on {task.deadline}" + 
                  (f" with {task.priority} priority" if task.priority else ""))
            time.sleep(0.5)  # Simulate API latency
            
            return ToolResult(
                success=True,
                message=f"Successfully scheduled task '{task.name}' on {task.deadline}",
                data={"task_id": f"task_{hash(task.name) % 10000}", "calendar_event_created": True}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Failed to schedule task: {str(e)}"
            )


# ================
# Notification Tool
# ================
class NotificationTool(BaseTool):
    """
    Tool for sending notifications about tasks.
    
    This tool can notify about unrealistic deadlines or other task-related concerns.
    """
    name: str = "notifier"
    description: str = "Sends notifications about tasks"
    
    def run(self, task: Task, message: Optional[str] = None) -> ToolResult:
        """
        Send a notification about a task.
        
        Args:
            task: The task to notify about
            message: Optional custom notification message
            
        Returns:
            ToolResult: The result of the notification operation
        """
        # In a real implementation, this might integrate with email, SMS, Slack, etc.
        # For demonstration purposes, we simulate it
        try:
            if not message:
                # Generate a default message based on task properties
                if not task.is_deadline_realistic():
                    message = f"⚠️ Warning: The deadline for '{task.name}' may be unrealistic."
                else:
                    message = f"Reminder: Task '{task.name}' is due on {task.deadline}."
            
            recipient = task.assignee if task.assignee else "Team"
            print(f"[Notification to {recipient}] {message}")
            time.sleep(0.3)  # Simulate notification latency
            
            return ToolResult(
                success=True,
                message=f"Successfully sent notification about task '{task.name}'",
                data={"recipient": recipient, "notification_sent": True}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Failed to send notification: {str(e)}"
            )


# ================
# Base Agent
# ================
class BaseAgent:
    """
    Base class for all AI-powered agents.
    
    Defines the interface and common functionality for agents.
    """
    name: str = "base_agent"
    description: str = "Base agent class that all agents derive from."
    
    def __init__(self):
        """Initialize the agent and register available tools."""
        self.tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool with the agent.
        
        Args:
            tool: The tool to register
        """
        self.tools[tool.name] = tool
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a registered tool by name.
        
        Args:
            tool_name: The name of the tool to retrieve
            
        Returns:
            The requested tool or None if not found
        """
        return self.tools.get(tool_name)
    
    def use_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Use a registered tool.
        
        Args:
            tool_name: The name of the tool to use
            **kwargs: Parameters to pass to the tool
            
        Returns:
            ToolResult: The result of the tool execution
            
        Raises:
            ValueError: If the requested tool is not registered
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                message=f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            )
        
        return tool.run(**kwargs)
    
    def process_text(self, text_input: str) -> Any:
        """
        Process text input to extract information or perform actions.
        
        Args:
            text_input: The text to process
            
        Returns:
            The result of processing the text
            
        Raises:
            NotImplementedError: This method must be implemented by derived classes
        """
        raise NotImplementedError("Derived agents must implement the 'process_text' method")


# ================
# Scheduler Agent
# ================
class SchedulerAgent(BaseAgent):
    """
    Agent that specializes in task scheduling.
    
    This agent uses LLM capabilities to extract tasks from text
    and then schedules them using the appropriate tools.
    """
    name: str = "scheduler_agent"
    description: str = "Agent that extracts and schedules tasks from text input"
    
    def __init__(self):
        """Initialize the scheduler agent and register scheduling tools."""
        super().__init__()
        
        # Register available tools
        self.register_tool(SchedulerTool())
        self.register_tool(NotificationTool())
    
    def process_text(self, text_input: str) -> Dict[str, Any]:
        """
        Process text input to extract tasks and schedule them.
        
        Args:
            text_input: The text to process
            
        Returns:
            A dictionary with processing results including:
            - summary: The text summary
            - tasks: The extracted tasks
            - tool_actions: The tool actions performed with results
        """
        print("\n------ Processing Text Input ------")
        
        # Get list of available tools for the LLM to choose from
        available_tools = list(self.tools.keys())
        
        # Use the LLM to extract tasks and determine which tools to use
        prompt = (
            "Please analyze the following text and extract:\n"
            "1. A concise summary of the content\n"
            "2. Any actionable tasks or to-dos mentioned\n"
            "3. Suggested deadlines for those tasks\n"
            "4. Choose which tools to use for each task\n\n"
            f"Text: {text_input}\n\n"
        )
        
        llm_response = LLMService.call(prompt, available_tools)
        
        # Output the summary
        print("\n------ AI Summary ------")
        print(llm_response.summary)
        
        # Process the tasks and execute the tool actions
        tool_actions_results = []
        
        if not llm_response.tasks_with_actions:
            print("\nNo tasks identified in the text.")
        else:
            print(f"\n------ Found {len(llm_response.tasks_with_actions)} Tasks ------")
            
            for i, task_with_actions in enumerate(llm_response.tasks_with_actions, 1):
                task = task_with_actions.task
                actions = task_with_actions.actions
                
                print(f"\nTask {i}: {task.name}")
                print(f"Due: {task.deadline}")
                if task.priority:
                    print(f"Priority: {task.priority}")
                if task.assignee:
                    print(f"Assigned to: {task.assignee}")
                
                print("\nActions selected by AI:")
                for action in actions:
                    print(f"- Tool: {action.tool_name}")
                    print(f"  Reason: {action.reason}")
                    if action.message:
                        print(f"  Message: {action.message}")
                    
                    # Execute the tool based on the LLM's decision
                    if action.tool_name == "notifier":
                        result = self.use_tool("notifier", task=task, message=action.message)
                    else:
                        result = self.use_tool(action.tool_name, task=task)
                    
                    tool_actions_results.append({
                        "task": task.model_dump(),
                        "tool": action.tool_name,
                        "reason": action.reason,
                        "result": result.model_dump()
                    })
        
        # Return the processing results
        return {
            "summary": llm_response.summary,
            "tasks": [task_with_actions.task.model_dump() for task_with_actions in llm_response.tasks_with_actions],
            "tool_actions": tool_actions_results
        }


# ================
# Demonstration
# ================
if __name__ == "__main__":
    # Sample text that simulates user input or extracted meeting notes
    sample_text = (
        "We met with the marketing team this morning to discuss "
        "next month's campaign. We need final approval on the budget "
        "by the end of the week, and a timeline set no later than early next week. "
        "The team also mentioned we should create a social media plan and must "
        "finalize the campaign message within the next 14 days. John should prepare "
        "the analytics report for review by tomorrow."
    )
    
    print("\n==== AI Agent Demo ====")
    print("Processing text input:", sample_text[:50] + "...")
    
    # Instantiate the agent
    agent = SchedulerAgent()
    
    # Process the text and execute identified tasks
    result = agent.process_text(sample_text)
    
    print("\n==== Demo Complete ====")
    print(f"Processed {len(result['tasks'])} tasks with {len(result['tool_actions'])} tool actions.")