# Copyright (c) Microsoft. All rights reserved.
"""
Microsoft Agent Framework - Core Agent Patterns

This module consolidates essential coding patterns for working with agents,
tools, context providers, middleware, and session management.

Patterns covered:
1. Basic Agent Creation and Execution
2. Function Tools with @tool decorator
3. Multi-Turn Conversations with AgentSession
4. Custom Context Providers for Memory
5. Chat Middleware for Request/Response Interception
6. Tool Approval Workflows
7. Provider Support (OpenAI, Azure OpenAI, Anthropic, etc.)
"""

import asyncio
from typing import Annotated, Any
from random import randint

from agent_framework import Agent, tool, AgentSession, ContextProvider, SessionContext, ChatMiddleware
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
from pydantic import Field


# ============================================================================
# PATTERN 1: BASIC AGENT CREATION
# ============================================================================
async def pattern_hello_agent():
    """
    Pattern: Create and run your first agent
    
    Demonstrates:
    - Instantiating FoundryChatClient with Azure credentials
    - Creating an Agent with instructions
    - Running agent in both streaming and non-streaming modes
    """
    client = FoundryChatClient(
        project_endpoint="https://your-project.services.ai.azure.com",
        model="gpt-4o",
        credential=AzureCliCredential(),
    )

    agent = Agent(
        client=client,
        name="HelloAgent",
        instructions="You are a friendly assistant. Keep your answers brief.",
    )

    # Non-streaming: get complete response at once
    result = await agent.run("What is the capital of France?")
    print(f"Non-streaming result: {result}")

    # Streaming: receive tokens as they are generated
    print("Streaming result: ", end="", flush=True)
    async for chunk in agent.run("Tell me a fun fact.", stream=True):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()


# ============================================================================
# PATTERN 2: FUNCTION TOOLS
# ============================================================================
@tool(approval_mode="never_require")  # Use "always_require" in production
def get_weather(
    location: Annotated[str, Field(description="The location to get weather for.")],
) -> str:
    """Get the weather for a given location.
    
    Use Annotated[type, Field(description=...)] for better documentation.
    """
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


@tool(approval_mode="never_require")
def calculate(
    operation: Annotated[str, Field(description="Math operation: +, -, *, /")],
    a: Annotated[float, Field(description="First number")],
    b: Annotated[float, Field(description="Second number")],
) -> float:
    """Perform basic math operations."""
    if operation == "+":
        return a + b
    elif operation == "-":
        return a - b
    elif operation == "*":
        return a * b
    elif operation == "/":
        return a / b if b != 0 else float("nan")
    else:
        raise ValueError(f"Unknown operation: {operation}")


async def pattern_agent_with_tools():
    """
    Pattern: Add function tools to your agent
    
    Demonstrates:
    - Defining tools with @tool decorator
    - Using Annotated types with Field descriptions
    - Passing tools to Agent constructor
    - Agent automatically calls tools when needed
    """
    client = FoundryChatClient(
        project_endpoint="https://your-project.services.ai.azure.com",
        model="gpt-4o",
        credential=AzureCliCredential(),
    )

    agent = Agent(
        client=client,
        name="WeatherAgent",
        instructions="You are a helpful weather agent. Use the get_weather tool to answer questions.",
        tools=[get_weather, calculate],
    )

    result = await agent.run("What's the weather in Seattle and what is 5 + 3?")
    print(f"Result with tools: {result}")


# ============================================================================
# PATTERN 3: MULTI-TURN CONVERSATIONS WITH SESSION
# ============================================================================
async def pattern_multi_turn_conversation():
    """
    Pattern: Maintain conversation history across turns
    
    Demonstrates:
    - Creating an AgentSession for stateful conversations
    - Passing session to multiple agent.run() calls
    - Agent automatically includes previous messages in context
    - Conversation state persists across turns
    """
    client = FoundryChatClient(
        project_endpoint="https://your-project.services.ai.azure.com",
        model="gpt-4o",
        credential=AzureCliCredential(),
    )

    agent = Agent(
        client=client,
        name="ChatAgent",
        instructions="You are a helpful assistant. Remember context from previous messages.",
    )

    # Create a session to maintain conversation history
    session = AgentSession(id="user-123")

    # Turn 1
    result1 = await agent.run("My name is Alice and I like Python", session=session)
    print(f"Turn 1: {result1}")

    # Turn 2 - Agent remembers from Turn 1
    result2 = await agent.run("What's my name?", session=session)
    print(f"Turn 2: {result2}")

    # Turn 3 - Full conversation history is available
    result3 = await agent.run("What did I say I like?", session=session)
    print(f"Turn 3: {result3}")

    # Serialize session state for storage
    session_dict = session.to_dict()
    
    # Restore session from serialized state
    restored_session = AgentSession.from_dict(session_dict)
    result4 = await agent.run("Summarize what you know about me", session=restored_session)
    print(f"Turn 4 (after restore): {result4}")


# ============================================================================
# PATTERN 4: CUSTOM CONTEXT PROVIDERS (MEMORY)
# ============================================================================
class UserMemoryProvider(ContextProvider):
    """A context provider that remembers user info in session state.
    
    Extends before_run and after_run hooks to:
    - Inject personalization before each agent call
    - Extract and store new information after each call
    """

    DEFAULT_SOURCE_ID = "user_memory"

    def __init__(self):
        super().__init__(self.DEFAULT_SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject personalization instructions based on stored user info."""
        user_name = state.get("user_name")
        if user_name:
            context.extend_instructions(
                self.source_id,
                f"The user's name is {user_name}. Always address them by name.",
            )
        else:
            context.extend_instructions(
                self.source_id,
                "You don't know the user's name yet. Ask for it politely.",
            )

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Extract and store user info in session state after each call."""
        for msg in context.input_messages:
            text = msg.text if hasattr(msg, "text") else ""
            if isinstance(text, str) and "my name is" in text.lower():
                # Extract name (naive example)
                parts = text.lower().split("my name is ")
                if len(parts) > 1:
                    name = parts[1].strip().split()[0]
                    state["user_name"] = name


class SessionVariableProvider(ContextProvider):
    """A context provider that injects session variables into instructions.
    
    Useful for:
    - Passing user preferences
    - Injecting context from external systems
    - Implementing persona variations
    """

    def __init__(self, variable_name: str):
        super().__init__(source_id=f"var_{variable_name}")
        self.variable_name = variable_name

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject variable value if present in session state."""
        value = state.get(self.variable_name)
        if value:
            context.extend_instructions(
                self.source_id,
                f"Current {self.variable_name}: {value}",
            )


async def pattern_context_providers():
    """
    Pattern: Use custom context providers for memory
    
    Demonstrates:
    - Creating custom ContextProvider subclass
    - Implementing before_run() and after_run() hooks
    - Accessing and modifying session state
    - Injecting dynamic instructions
    """
    client = FoundryChatClient(
        project_endpoint="https://your-project.services.ai.azure.com",
        model="gpt-4o",
        credential=AzureCliCredential(),
    )

    session = AgentSession(id="user-456")
    session.state = {}

    agent = Agent(
        client=client,
        name="MemoryAgent",
        instructions="You are a helpful assistant with memory.",
        context_providers=[
            UserMemoryProvider(),
            SessionVariableProvider("user_preference"),
        ],
    )

    # First interaction
    result1 = await agent.run("My name is Bob and I prefer detailed answers.", session=session)
    print(f"Result 1: {result1}")

    # Second interaction - agent remembers Bob's name
    result2 = await agent.run("What do you know about me?", session=session)
    print(f"Result 2: {result2}")


# ============================================================================
# PATTERN 5: CHAT MIDDLEWARE
# ============================================================================
class LoggingMiddleware(ChatMiddleware):
    """A middleware that logs all messages and responses.
    
    Use cases:
    - Audit trails
    - Performance monitoring
    - Request/response validation
    """

    async def on_chat_complete(self, messages, response, **kwargs):
        """Called after model completes response."""
        print(f"[LOG] User asked: {messages[-1].text if hasattr(messages[-1], 'text') else '...'}")
        print(f"[LOG] Model responded: {response.text[:100]}...")
        return response  # Return response unchanged


class RetryMiddleware(ChatMiddleware):
    """A middleware that retries on specific errors.
    
    Use cases:
    - Handle rate limiting
    - Recover from transient failures
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def on_chat_error(self, exception, **kwargs):
        """Called when chat encounter an error."""
        if "rate limit" in str(exception).lower():
            print(f"[RETRY] Rate limited, retrying...")
            # Retry logic would go here
            return None  # None signals retry
        raise exception


class TokenCountingMiddleware(ChatMiddleware):
    """A middleware that counts tokens used.
    
    Use cases:
    - Cost tracking
    - Budget monitoring
    - Usage analytics
    """

    def __init__(self):
        self.total_tokens = 0

    async def on_chat_complete(self, messages, response, **kwargs):
        """Count tokens in response."""
        # Simplified token counting (would need tiktoken in production)
        token_count = len(response.text.split()) * 1.3  # Approximate
        self.total_tokens += token_count
        print(f"[TOKENS] Estimated tokens: {token_count:.0f}, Total: {self.total_tokens:.0f}")
        return response


async def pattern_middleware():
    """
    Pattern: Use middleware for cross-cutting concerns
    
    Demonstrates:
    - Implementing ChatMiddleware subclass
    - Hooking into on_chat_complete, on_chat_error
    - Chaining multiple middleware
    - Implementing logging, retry, token counting
    """
    client = FoundryChatClient(
        project_endpoint="https://your-project.services.ai.azure.com",
        model="gpt-4o",
        credential=AzureCliCredential(),
    )

    # Add multiple middleware
    middleware = [
        LoggingMiddleware(),
        TokenCountingMiddleware(),
        RetryMiddleware(max_retries=3),
    ]

    agent = Agent(
        client=client,
        name="MiddlewareAgent",
        instructions="You are a helpful assistant.",
        middleware=middleware,
    )

    result = await agent.run("What is 2 + 2?")
    print(f"Result: {result}")


# ============================================================================
# PATTERN 6: TOOL APPROVAL WORKFLOWS
# ============================================================================
@tool(approval_mode="always_require")
def delete_file(
    filepath: Annotated[str, Field(description="Path to file to delete")],
) -> str:
    """Delete a file - always requires user approval."""
    return f"Would delete: {filepath}"


@tool(approval_mode="never_require")
def read_file(
    filepath: Annotated[str, Field(description="Path to file to read")],
) -> str:
    """Read a file - no approval needed."""
    return f"File contents of {filepath}..."


async def pattern_tool_approval():
    """
    Pattern: Implement tool approval workflows
    
    Demonstrates:
    - approval_mode="always_require" for sensitive operations
    - approval_mode="never_require" for safe operations
    - approval_mode="sometimes_require" for context-based approval
    
    The framework handles user prompts automatically.
    """
    client = FoundryChatClient(
        project_endpoint="https://your-project.services.ai.azure.com",
        model="gpt-4o",
        credential=AzureCliCredential(),
    )

    agent = Agent(
        client=client,
        name="FileAgent",
        instructions="You can read and delete files.",
        tools=[read_file, delete_file],
    )

    # User will be prompted to approve delete_file calls
    result = await agent.run("Delete the file config.json")
    print(f"Result: {result}")


# ============================================================================
# PATTERN 7: MULTIPLE PROVIDER SUPPORT
# ============================================================================
async def pattern_provider_openai():
    """Pattern: Use OpenAI provider instead of Azure Foundry."""
    from agent_framework.openai import OpenAIChatClient
    
    client = OpenAIChatClient(
        api_key="sk-...",
        model="gpt-4o",
    )

    agent = Agent(
        client=client,
        name="OpenAIAgent",
        instructions="You are a helpful assistant.",
    )

    result = await agent.run("Hello!")
    print(f"Result: {result}")


async def pattern_provider_anthropic():
    """Pattern: Use Anthropic provider (Claude)."""
    from agent_framework.anthropic import AnthropicChatClient
    
    client = AnthropicChatClient(
        api_key="sk-ant-...",
        model="claude-3-opus",
    )

    agent = Agent(
        client=client,
        name="ClaudeAgent",
        instructions="You are a helpful assistant.",
    )

    result = await agent.run("Hello!")
    print(f"Result: {result}")


# ============================================================================
# PATTERN 8: COMBINING PATTERNS - FULL EXAMPLE
# ============================================================================
class ContextAwareProvider(ContextProvider):
    """Inject context based on user profile."""
    
    def __init__(self):
        super().__init__("context_aware")

    async def before_run(self, *, agent, session, context, state, **kwargs):
        user_role = state.get("user_role", "guest")
        context.extend_instructions(
            self.source_id,
            f"The user has the role: {user_role}. Adjust your tone accordingly.",
        )


async def pattern_combined_example():
    """
    Pattern: Combine multiple patterns in a real-world scenario
    
    Demonstrates:
    - Agent with tools
    - Session management
    - Custom context provider
    - Middleware
    """
    client = FoundryChatClient(
        project_endpoint="https://your-project.services.ai.azure.com",
        model="gpt-4o",
        credential=AzureCliCredential(),
    )

    session = AgentSession(id="user-789")
    session.state = {"user_role": "admin"}

    agent = Agent(
        client=client,
        name="FullFeaturedAgent",
        instructions="You are a multi-purpose assistant.",
        tools=[get_weather, calculate],
        context_providers=[ContextAwareProvider()],
        middleware=[LoggingMiddleware()],
    )

    # Multi-turn conversation with tools
    result1 = await agent.run("What's the weather in Paris?", session=session)
    print(f"Turn 1: {result1}")

    result2 = await agent.run("Calculate 10 * 5 for me", session=session)
    print(f"Turn 2: {result2}")

    result3 = await agent.run("Remember to be formal since I'm an admin", session=session)
    print(f"Turn 3: {result3}")


if __name__ == "__main__":
    # Run examples (comment out as needed)
    print("This module contains reusable patterns. Import and use as needed.")
    print("Examples are provided in async functions.")
