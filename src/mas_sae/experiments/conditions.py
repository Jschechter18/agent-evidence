from mas_sae.agents.critic import CriticCondition


OMITTED = object()


def resolve_conditions(values=OMITTED):
    """Omission preserves V1; explicit lists must be nonempty and unique."""
    if values is OMITTED:
        return tuple(CriticCondition)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("active_conditions must be a nonempty list")
    conditions = tuple(CriticCondition(value) for value in values)
    if len(set(conditions)) != len(conditions):
        raise ValueError("active_conditions must be unique")
    return conditions
