"""Advisor agent — weekly financial briefing synthesised from all agents."""

__all__ = ["AdvisorAgent"]


def __getattr__(name: str):
    if name == "AdvisorAgent":
        from services.agents.advisor.agent import AdvisorAgent
        return AdvisorAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
