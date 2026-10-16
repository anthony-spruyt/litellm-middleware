from litellm.integrations.custom_guardrail import CustomGuardrail


class ToolGuardPlaceholder(CustomGuardrail):
    """Registers the `tool-guard` guardrail so the Guardrails Monitor detail page resolves.

    Defines no hooks: the base class passes requests through and writes no record, even when a caller opts in.
    The record itself comes from the tool-guard middleware.
    """
