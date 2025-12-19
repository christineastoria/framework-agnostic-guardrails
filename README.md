# Framework-Agnostic Guardrails

A simple, framework-agnostic wrapper for adding guardrails to AI agents at the application level.

## Philosophy

We believe **application-level guardrails are more effective** than placing guardrails in the observability layer. Security and compliance decisions should be made where they matter most - in your application code, before and after your agent runs.

Inspired by [LangChain's guardrails middleware](https://docs.langchain.com/oss/python/langchain/guardrails), but designed to work with **any agent framework**.

This project demonstrates how easy it is to integrate guardrails into your agent using open-source tools like [OpenEvals](https://github.com/langchain-ai/openevals).

## Quick Start

### Installation

Using [UV](https://docs.astral.sh/uv/) (recommended):

```bash
# Install dependencies
uv sync

# Set up your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Basic Usage

```python
from guard import Guard, PIIDetectionGuardrail, JailbreakGuardrail

# Create guard with multiple guardrails
guard = Guard([
    PIIDetectionGuardrail(),     # Pattern-based
    JailbreakGuardrail(),         # LLM-based
])

# Works with ANY agent function
def my_agent(user_input: str) -> str:
    # Your agent logic (OpenAI, Anthropic, whatever)
    return client.chat.completions.create(...)

# Run with guardrails
result = guard.run(
    user_input="What's the weather?",
    agent_call=my_agent
)

if result.blocked:
    print(f"Blocked: {result.findings}")
else:
    print(f"Safe: {result.output}")
```

### Flexible Input/Output Validation

Each guardrail can be configured to check inputs, outputs, or both:

```python
guard = Guard([
    # Check only user input for jailbreak attempts
    JailbreakGuardrail(check_input=True, check_output=False),
    
    # Check only agent output for compliance
    RegulatoryComplianceGuardrail(check_input=False, check_output=True),
    
    # Check both input and output for PII
    PIIDetectionGuardrail(check_input=True, check_output=True),
])
```

**Smart defaults:** Each guardrail comes with sensible defaults based on its purpose (e.g., jailbreak checks input only, compliance checks output only).

## Available Guardrails

### Pattern-Based (Fast, No API Required)
- **PIIDetectionGuardrail** - Block emails, SSNs, phone numbers
- **ToxicLanguageGuardrail** - Block offensive content

### LLM-Based (Semantic, Uses OpenEvals)
- **JailbreakGuardrail** - Detect manipulation attempts
- **RegulatoryComplianceGuardrail** - Ensure financial/legal compliance
- **BiasDetectionGuardrail** - Catch discriminatory content
- **CorrectnessGuardrail** - Validate factual accuracy
- **HallucinationGuardrail** - Detect made-up information

### Custom
- **CustomGuardrail** - Your own business logic

## Examples

See [`examples.py`](examples.py) for detailed demonstrations of each guardrail, including:
- Input vs output validation
- Pattern-based vs LLM-based checks
- Custom business logic
- Multiple guardrails working together

## Credits

Built using:
- [OpenEvals](https://github.com/langchain-ai/openevals) - LLM-as-judge framework
- Inspired by [LangChain's guardrails middleware](https://docs.langchain.com/oss/python/langchain/guardrails)

## License

MIT
