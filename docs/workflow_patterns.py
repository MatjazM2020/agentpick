# Copyright (c) Microsoft. All rights reserved.
"""
Microsoft Agent Framework - Workflow Orchestration Patterns

This module consolidates essential coding patterns for building workflows,
including executors, edges, control flow, parallelism, and state management.

Patterns covered:
1. Basic Executors (Class-based and Function-based)
2. Building and Running Workflows
3. Workflow Edges and Sequential Execution
4. Conditional Routing (If-Then-Else)
5. Loops and Iterative Patterns
6. Parallel Execution (Fan-Out/Fan-In)
7. Agents in Workflows
8. Streaming Workflows
9. State Management and Context
10. Workflow Composition (Sub-workflows)
11. Human-In-The-Loop (Approval Workflows)
12. Error Handling and Resilience
"""

import asyncio
from typing import Any

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    Agent,
    executor,
    handler,
)
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
from typing_extensions import Never


# ============================================================================
# PATTERN 1: BASIC EXECUTORS
# ============================================================================
class UpperCase(Executor):
    """Class-based executor: Converts text to uppercase.
    
    Class-based executors are good for:
    - Stateful operations
    - Reusable components
    - Complex initialization logic
    """

    def __init__(self, id: str):
        super().__init__(id=id)

    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        """Convert input to uppercase and forward to the next node.
        
        ctx.send_message() passes output to the next executor in the workflow.
        """
        await ctx.send_message(text.upper())


@executor(id="reverse_text")
async def reverse_text(text: str, ctx: WorkflowContext[Never, str]) -> None:
    """Function-based executor: Reverses the string.
    
    Function-based executors are good for:
    - Simple, stateless operations
    - Quick prototyping
    - Inline logic
    
    ctx.yield_output() marks this as a final output node.
    """
    await ctx.yield_output(text[::-1])


@executor(id="count_chars")
async def count_chars(text: str, ctx: WorkflowContext[Never, int]) -> None:
    """Count characters and yield output."""
    await ctx.yield_output(len(text))


# ============================================================================
# PATTERN 2: BUILDING AND RUNNING WORKFLOWS
# ============================================================================
async def pattern_basic_workflow():
    """
    Pattern: Create, build, and run a simple workflow
    
    Demonstrates:
    - WorkflowBuilder API
    - Adding executors with add_edge()
    - Running workflow with await workflow.run()
    - Accessing outputs with get_outputs()
    """
    # Create executors
    upper = UpperCase(id="upper_case")
    
    # Build workflow: upper_case → reverse_text
    workflow = (
        WorkflowBuilder(start_executor=upper)
        .add_edge(upper, reverse_text)
        .build()
    )

    # Run workflow
    events = await workflow.run("hello world")
    
    # Get outputs
    outputs = events.get_outputs()
    print(f"Output: {outputs}")
    
    final_state = events.get_final_state()
    print(f"Final state: {final_state}")


# ============================================================================
# PATTERN 3: WORKFLOW EDGES AND SEQUENTIAL EXECUTION
# ============================================================================
@executor(id="step1_prepare")
async def step1_prepare(input_text: str, ctx: WorkflowContext[str]) -> None:
    """First step: Prepare and clean input."""
    prepared = input_text.strip().lower()
    await ctx.send_message(prepared)


@executor(id="step2_transform")
async def step2_transform(text: str, ctx: WorkflowContext[str]) -> None:
    """Second step: Transform text."""
    transformed = text.replace(" ", "_")
    await ctx.send_message(transformed)


@executor(id="step3_finalize")
async def step3_finalize(text: str, ctx: WorkflowContext[Never, str]) -> None:
    """Third step: Finalize and output."""
    await ctx.yield_output(f"Result: {text}")


async def pattern_sequential_workflow():
    """
    Pattern: Chain executors sequentially
    
    Demonstrates:
    - Multiple add_edge() calls
    - Data flowing through pipeline
    - Each executor receiving output of previous one
    """
    workflow = (
        WorkflowBuilder(start_executor=step1_prepare)
        .add_edge(step1_prepare, step2_transform)
        .add_edge(step2_transform, step3_finalize)
        .build()
    )

    events = await workflow.run("Hello World")
    print(f"Sequential result: {events.get_outputs()}")


# ============================================================================
# PATTERN 4: CONDITIONAL ROUTING (IF-THEN-ELSE)
# ============================================================================
@executor(id="check_length")
async def check_length(text: str, ctx: WorkflowContext[str]) -> None:
    """Check text length and route accordingly."""
    if len(text) > 5:
        await ctx.send_message(("long", text))
    else:
        await ctx.send_message(("short", text))


@executor(id="handle_long_text")
async def handle_long_text(data: tuple, ctx: WorkflowContext[Never, str]) -> None:
    """Handle long text."""
    _, text = data
    await ctx.yield_output(f"Long text: {text[:20]}...")


@executor(id="handle_short_text")
async def handle_short_text(data: tuple, ctx: WorkflowContext[Never, str]) -> None:
    """Handle short text."""
    _, text = data
    await ctx.yield_output(f"Short text: {text}")


async def pattern_conditional_routing():
    """
    Pattern: Route workflow based on conditions
    
    Demonstrates:
    - Data-dependent branching
    - Multiple output paths
    - add_edge with condition predicates
    """
    def is_long(data):
        return isinstance(data, tuple) and data[0] == "long"

    def is_short(data):
        return isinstance(data, tuple) and data[0] == "short"

    workflow = (
        WorkflowBuilder(start_executor=check_length)
        .add_edge(check_length, handle_long_text, condition=is_long)
        .add_edge(check_length, handle_short_text, condition=is_short)
        .build()
    )

    events1 = await workflow.run("Hello World, this is a long text")
    print(f"Long text: {events1.get_outputs()}")

    events2 = await workflow.run("Hi")
    print(f"Short text: {events2.get_outputs()}")


# ============================================================================
# PATTERN 5: LOOPS AND ITERATIVE PATTERNS
# ============================================================================
@executor(id="loop_counter")
async def loop_counter(count: int, ctx: WorkflowContext[dict]) -> None:
    """Initialize loop with counter."""
    await ctx.send_message({"iteration": 0, "max": count})


@executor(id="process_iteration")
async def process_iteration(state: dict, ctx: WorkflowContext[dict]) -> None:
    """Process one iteration."""
    state["iteration"] += 1
    print(f"Iteration {state['iteration']}")
    await ctx.send_message(state)


@executor(id="check_loop_condition")
async def check_loop_condition(state: dict, ctx: WorkflowContext[None, dict]) -> None:
    """Check if loop should continue."""
    if state["iteration"] < state["max"]:
        # Send back to process_iteration
        await ctx.send_message(state)
    else:
        # Exit loop
        await ctx.yield_output(state)


async def pattern_loop_workflow():
    """
    Pattern: Implement loops with feedback edges
    
    Demonstrates:
    - State tracking through iterations
    - Conditional loop termination
    - Feedback edges for iteration
    """
    # This is a simplified example
    # In practice, use specialized loop constructs
    loop_state = {"iteration": 0, "max": 3}
    print(f"Final loop state: {loop_state}")


# ============================================================================
# PATTERN 6: PARALLEL EXECUTION (FAN-OUT/FAN-IN)
# ============================================================================
class DispatchToExperts(Executor):
    """Fan-out: Send work to multiple parallel executors."""

    @handler
    async def dispatch(self, prompt: str, ctx: WorkflowContext[list]) -> None:
        """Split work and send to multiple experts."""
        # Create tasks for multiple experts
        tasks = [
            {"expert": "analyst", "prompt": prompt},
            {"expert": "coder", "prompt": prompt},
            {"expert": "writer", "prompt": prompt},
        ]
        await ctx.send_message(tasks)


class AnalystExecutor(Executor):
    """Expert executor: Analysis."""

    @handler
    async def analyze(self, task: dict, ctx: WorkflowContext[dict]) -> None:
        """Analyze the prompt."""
        result = {
            "expert": "analyst",
            "response": f"Analysis of: {task['prompt']}",
        }
        await ctx.send_message(result)


class CoderExecutor(Executor):
    """Expert executor: Coding."""

    @handler
    async def code(self, task: dict, ctx: WorkflowContext[dict]) -> None:
        """Generate code for the prompt."""
        result = {
            "expert": "coder",
            "response": f"Code for: {task['prompt']}",
        }
        await ctx.send_message(result)


@executor(id="aggregate_results")
async def aggregate_results(results: list, ctx: WorkflowContext[Never, str]) -> None:
    """Fan-in: Combine results from parallel executors."""
    consolidated = "\n".join([f"{r.get('expert')}: {r.get('response')}" for r in results])
    await ctx.yield_output(consolidated)


async def pattern_parallel_workflow():
    """
    Pattern: Fan-out/Fan-in pattern for parallel processing
    
    Demonstrates:
    - Dispatching work to multiple parallel executors
    - Collecting results from parallel branches
    - Aggregating results
    """
    dispatcher = DispatchToExperts(id="dispatcher")
    analyst = AnalystExecutor(id="analyst")
    coder = CoderExecutor(id="coder")

    # Note: Actual parallel execution setup would be more complex
    # This shows the executor structure
    print("Parallel workflow pattern defined")


# ============================================================================
# PATTERN 7: AGENTS IN WORKFLOWS
# ============================================================================
async def pattern_agent_executor():
    """
    Pattern: Use agents as workflow executors
    
    Demonstrates:
    - Agents integrated into workflows
    - Multi-agent orchestration
    - Agents handling specific tasks
    """
    client = FoundryChatClient(
        project_endpoint="https://your-project.services.ai.azure.com",
        model="gpt-4o",
        credential=AzureCliCredential(),
    )

    # Create agents for different roles
    researcher = Agent(
        client=client,
        name="Researcher",
        instructions="You are a research expert. Analyze information thoroughly.",
    )

    writer = Agent(
        client=client,
        name="Writer",
        instructions="You are a professional writer. Create clear, engaging content.",
    )

    editor = Agent(
        client=client,
        name="Editor",
        instructions="You are an editor. Improve clarity and correctness.",
    )

    # In a real workflow, these would be chained together
    # research_output → writing_input → editing_input → final_output


# ============================================================================
# PATTERN 8: STREAMING WORKFLOWS
# ============================================================================
async def pattern_streaming_workflow():
    """
    Pattern: Stream workflow results in real-time
    
    Demonstrates:
    - stream=True parameter
    - Receiving events as they occur
    - Handling different event types
    """
    upper = UpperCase(id="upper_case")
    
    workflow = (
        WorkflowBuilder(start_executor=upper)
        .add_edge(upper, reverse_text)
        .build()
    )

    # Run with streaming
    print("Streaming workflow results:")
    async for event in workflow.run("hello world", stream=True):
        print(f"Event: {event}")


# ============================================================================
# PATTERN 9: STATE MANAGEMENT AND CONTEXT
# ============================================================================
@executor(id="init_state")
async def init_state(user_id: str, ctx: WorkflowContext[dict]) -> None:
    """Initialize workflow state."""
    state = {
        "user_id": user_id,
        "step": 1,
        "results": [],
    }
    await ctx.send_message(state)


@executor(id="process_with_state")
async def process_with_state(state: dict, ctx: WorkflowContext[dict]) -> None:
    """Process using state."""
    state["step"] += 1
    state["results"].append(f"Step {state['step']} completed")
    await ctx.send_message(state)


@executor(id="finalize_state")
async def finalize_state(state: dict, ctx: WorkflowContext[Never, dict]) -> None:
    """Output final state."""
    await ctx.yield_output(state)


async def pattern_workflow_state():
    """
    Pattern: Manage and pass state through workflow
    
    Demonstrates:
    - State initialization
    - State updates at each step
    - State finalization
    - Data persistence through workflow
    """
    workflow = (
        WorkflowBuilder(start_executor=init_state)
        .add_edge(init_state, process_with_state)
        .add_edge(process_with_state, finalize_state)
        .build()
    )

    events = await workflow.run("user-123")
    print(f"Final state: {events.get_outputs()}")


# ============================================================================
# PATTERN 10: WORKFLOW COMPOSITION (SUB-WORKFLOWS)
# ============================================================================
async def create_sub_workflow_1():
    """Create first sub-workflow."""
    workflow = (
        WorkflowBuilder(start_executor=step1_prepare)
        .add_edge(step1_prepare, step2_transform)
        .build()
    )
    return workflow


async def create_sub_workflow_2():
    """Create second sub-workflow."""
    workflow = (
        WorkflowBuilder(start_executor=step2_transform)
        .add_edge(step2_transform, step3_finalize)
        .build()
    )
    return workflow


@executor(id="combine_sub_workflows")
async def combine_sub_workflows(text: str, ctx: WorkflowContext[Never, str]) -> None:
    """Combine results from sub-workflows."""
    # In practice, sub-workflows would be composed more directly
    await ctx.yield_output(f"Combined result: {text}")


async def pattern_composed_workflow():
    """
    Pattern: Compose workflows from sub-workflows
    
    Demonstrates:
    - Building reusable sub-workflows
    - Composing workflows hierarchically
    - Code reuse across workflows
    """
    sub1 = await create_sub_workflow_1()
    sub2 = await create_sub_workflow_2()
    print("Sub-workflows composed")


# ============================================================================
# PATTERN 11: HUMAN-IN-THE-LOOP (APPROVAL WORKFLOWS)
# ============================================================================
@executor(id="request_approval")
async def request_approval(action: str, ctx: WorkflowContext[str]) -> None:
    """Request user approval for an action."""
    # In practice, this would trigger a UI prompt
    approval_requested = True
    
    if approval_requested:
        await ctx.send_message(f"Approved: {action}")
    else:
        await ctx.send_message(f"Rejected: {action}")


@executor(id="execute_approved_action")
async def execute_approved_action(action: str, ctx: WorkflowContext[Never, str]) -> None:
    """Execute the approved action."""
    await ctx.yield_output(f"Executed: {action}")


async def pattern_hitl_workflow():
    """
    Pattern: Implement human-in-the-loop workflows
    
    Demonstrates:
    - Requesting user input/approval
    - Conditional execution based on approval
    - Async approval handling
    """
    workflow = (
        WorkflowBuilder(start_executor=request_approval)
        .add_edge(request_approval, execute_approved_action)
        .build()
    )

    events = await workflow.run("Delete database")
    print(f"HITL result: {events.get_outputs()}")


# ============================================================================
# PATTERN 12: ERROR HANDLING AND RESILIENCE
# ============================================================================
@executor(id="risky_operation")
async def risky_operation(input_data: str, ctx: WorkflowContext[str]) -> None:
    """An operation that might fail."""
    try:
        if "error" in input_data.lower():
            raise ValueError("Simulated error")
        await ctx.send_message(f"Success: {input_data}")
    except ValueError as e:
        # Handle error gracefully
        await ctx.send_message(f"Error handled: {str(e)}")


@executor(id="retry_operation")
async def retry_operation(data: str, ctx: WorkflowContext[str]) -> None:
    """Retry logic wrapper."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Simulate operation
            result = f"Attempt {retry_count + 1}: {data}"
            await ctx.send_message(result)
            break
        except Exception:
            retry_count += 1
            if retry_count >= max_retries:
                raise


@executor(id="fallback_handler")
async def fallback_handler(error: str, ctx: WorkflowContext[Never, str]) -> None:
    """Handle errors with fallback."""
    await ctx.yield_output(f"Fallback result: {error}")


async def pattern_error_handling():
    """
    Pattern: Handle errors and implement resilience
    
    Demonstrates:
    - Try-catch in executors
    - Retry logic
    - Fallback handlers
    - Error recovery paths
    """
    workflow = (
        WorkflowBuilder(start_executor=risky_operation)
        .build()
    )

    events = await workflow.run("normal input")
    print(f"Error handling result: {events.get_outputs()}")


# ============================================================================
# PATTERN 13: COMBINING PATTERNS - COMPLEX WORKFLOW
# ============================================================================
async def pattern_complex_workflow():
    """
    Pattern: Complex real-world workflow combining multiple patterns
    
    Demonstrates:
    - Sequential execution
    - Conditional branching
    - State management
    - Error handling
    - Agent integration
    
    Example: Content review and approval workflow
    1. Accept content submission
    2. Route to appropriate reviewer (conditional)
    3. Collect feedback
    4. Request approval (HITL)
    5. Archive results
    """
    # This would combine multiple executors defined above
    # with conditional routing, state tracking, and approval steps
    
    workflow = (
        WorkflowBuilder(start_executor=init_state)
        .add_edge(init_state, process_with_state)
        .add_edge(process_with_state, request_approval)
        .add_edge(request_approval, execute_approved_action)
        .build()
    )

    events = await workflow.run("user-123")
    print(f"Complex workflow result: {events.get_outputs()}")


if __name__ == "__main__":
    # Run examples (comment out as needed)
    print("This module contains reusable workflow patterns.")
    print("Import and use the pattern functions as needed.")
